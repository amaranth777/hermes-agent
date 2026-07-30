"""Customization-aware git update analysis and source mutation.

The dashboard uses this module as the single authority for deciding whether a
checkout can safely follow the official Hermes ``main`` branch.  Analysis is
network-free; callers may fetch the configured official remote first.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_OFFICIAL_REPO = "github.com/nousresearch/hermes-agent"


class UpdateBlocked(RuntimeError):
    """The checkout cannot be updated without risking local work."""


@dataclass(frozen=True)
class RepositoryUpdateState:
    branch: str | None
    current_sha: str | None
    upstream_ref: str | None
    upstream_remote: str | None
    dirty: bool
    behind: int | None
    custom_commit_count: int
    custom_commits: list[dict[str, str]]
    diverged: bool
    strategy: str
    can_apply: bool
    blocking_reasons: list[str]
    backup_required: bool
    rebase_conflict: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UpdateResult:
    changed: bool
    strategy: str
    old_sha: str
    new_sha: str
    backup_ref: str | None


def _run(repo: Path, *args: str, check: bool = False, timeout: int = 15) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=check,
        timeout=timeout,
    )


def _official_url(url: str) -> bool:
    normalized = url.strip().lower().removesuffix(".git").rstrip("/")
    normalized = normalized.replace("git@github.com:", "github.com/")
    for prefix in ("https://", "http://", "ssh://git@", "git://"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    return normalized == _OFFICIAL_REPO


def resolve_official_remote(repo: Path) -> tuple[str | None, str | None]:
    """Return ``(remote, remote/main ref)`` only for the official repository."""
    for remote in ("upstream", "origin"):
        url = _run(repo, "remote", "get-url", remote)
        if url.returncode != 0 or not _official_url(url.stdout):
            continue
        ref = f"refs/remotes/{remote}/main"
        if _run(repo, "rev-parse", "--verify", ref).returncode == 0:
            return remote, f"{remote}/main"
    return None, None


def fetch_official_remote(repo: Path, remote: str, *, timeout: int = 30) -> None:
    result = _run(repo, "fetch", remote, "main", timeout=timeout)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        raise UpdateBlocked(detail[0] if detail else f"failed to fetch {remote}/main")


def _count(repo: Path, range_spec: str) -> int | None:
    result = _run(repo, "rev-list", "--count", range_spec)
    try:
        return int(result.stdout.strip()) if result.returncode == 0 else None
    except ValueError:
        return None


def _custom_commits(repo: Path, upstream_ref: str) -> list[dict[str, str]]:
    result = _run(repo, "log", "--format=%H%x1f%s", f"{upstream_ref}..HEAD")
    if result.returncode != 0:
        return []
    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        if "\x1f" not in line:
            continue
        sha, summary = line.split("\x1f", 1)
        rows.append({"sha": sha[:12], "summary": summary})
    return rows


def analyze_repository(repo: Path | str) -> RepositoryUpdateState:
    repo = Path(repo)
    sha_result = _run(repo, "rev-parse", "HEAD")
    current_sha = sha_result.stdout.strip() if sha_result.returncode == 0 else None
    branch_result = _run(repo, "symbolic-ref", "--quiet", "--short", "HEAD")
    branch = branch_result.stdout.strip() if branch_result.returncode == 0 else None
    dirty_result = _run(repo, "status", "--porcelain=v1", "--untracked-files=normal")
    dirty = dirty_result.returncode != 0 or bool(dirty_result.stdout.strip())
    remote, upstream_ref = resolve_official_remote(repo)

    reasons: list[str] = []
    if current_sha is None:
        reasons.append("not_a_git_checkout")
    if branch is None and current_sha is not None:
        reasons.append("detached_head")
    if upstream_ref is None:
        reasons.append("official_remote_missing")

    behind = _count(repo, f"HEAD..{upstream_ref}") if current_sha and upstream_ref else None
    custom_commits = _custom_commits(repo, upstream_ref) if current_sha and upstream_ref else []
    custom_count = len(custom_commits)
    diverged = bool((behind or 0) > 0 and custom_count > 0)

    if behind is None and upstream_ref is not None:
        reasons.append("history_unavailable")
    if (behind or 0) > 0 and dirty:
        reasons.append("dirty_worktree")

    if reasons:
        strategy = "blocked"
        can_apply = False
    elif behind == 0:
        strategy = "none"
        can_apply = False
    elif custom_count:
        strategy = "rebase-customizations"
        can_apply = True
    else:
        strategy = "fast-forward"
        can_apply = True

    return RepositoryUpdateState(
        branch=branch,
        current_sha=current_sha,
        upstream_ref=upstream_ref,
        upstream_remote=remote,
        dirty=dirty,
        behind=behind,
        custom_commit_count=custom_count,
        custom_commits=custom_commits,
        diverged=diverged,
        strategy=strategy,
        can_apply=can_apply,
        blocking_reasons=reasons,
        backup_required=can_apply,
    )


def _merge_base(repo: Path, upstream_ref: str) -> str:
    result = _run(repo, "merge-base", "HEAD", upstream_ref)
    if result.returncode != 0 or not result.stdout.strip():
        raise UpdateBlocked("official history has no merge base with this checkout")
    return result.stdout.strip()


def rebase_preflight(repo: Path | str, state: RepositoryUpdateState | None = None) -> bool:
    """Return whether custom commits replay cleanly without touching the checkout."""
    repo = Path(repo)
    state = state or analyze_repository(repo)
    if state.strategy != "rebase-customizations" or not state.current_sha or not state.upstream_ref:
        return True
    base = _merge_base(repo, state.upstream_ref)
    temp_parent = Path(tempfile.mkdtemp(prefix="hermes-update-preflight-"))
    worktree = temp_parent / "worktree"
    added = False
    try:
        add = _run(repo, "worktree", "add", "--detach", str(worktree), state.current_sha, timeout=30)
        if add.returncode != 0:
            raise UpdateBlocked("could not create update preflight worktree")
        added = True
        replay = _run(worktree, "rebase", "--onto", state.upstream_ref, base, timeout=120)
        if replay.returncode != 0:
            _run(worktree, "rebase", "--abort", timeout=30)
            return False
        return True
    finally:
        if added:
            _run(repo, "worktree", "remove", "--force", str(worktree), timeout=30)
            _run(repo, "worktree", "prune", timeout=30)
        shutil.rmtree(temp_parent, ignore_errors=True)


def analyze_with_preflight(repo: Path | str) -> RepositoryUpdateState:
    state = analyze_repository(repo)
    if state.strategy != "rebase-customizations" or not state.can_apply:
        return state
    if rebase_preflight(repo, state):
        return state
    return RepositoryUpdateState(
        **{
            **state.to_dict(),
            "strategy": "blocked",
            "can_apply": False,
            "blocking_reasons": [*state.blocking_reasons, "rebase_conflict"],
            "rebase_conflict": True,
            "backup_required": False,
        }
    )


def _backup_ref(repo: Path, sha: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"refs/hermes-backup/{stamp}-{sha[:12]}"
    ref = base
    suffix = 1
    while _run(repo, "show-ref", "--verify", "--quiet", ref).returncode == 0:
        suffix += 1
        ref = f"{base}-{suffix}"
    _run(repo, "update-ref", ref, sha, check=True)
    return ref


def apply_customization_update(repo: Path | str, *, fetch: bool = True) -> UpdateResult:
    """Safely advance to official main while preserving local commits."""
    repo = Path(repo)
    initial = analyze_repository(repo)
    if fetch:
        if not initial.upstream_remote:
            raise UpdateBlocked("official remote missing")
        fetch_official_remote(repo, initial.upstream_remote)
    state = analyze_repository(repo)
    if not state.can_apply or not state.current_sha or not state.upstream_ref:
        reasons = ", ".join(state.blocking_reasons) or "no official update available"
        raise UpdateBlocked(reasons)
    if state.strategy == "rebase-customizations" and not rebase_preflight(repo, state):
        raise UpdateBlocked("customization rebase conflict; checkout left unchanged")

    old_sha = state.current_sha
    backup_ref = _backup_ref(repo, old_sha)
    try:
        if state.strategy == "fast-forward":
            result = _run(repo, "merge", "--ff-only", state.upstream_ref, timeout=120)
        else:
            base = _merge_base(repo, state.upstream_ref)
            result = _run(repo, "rebase", "--onto", state.upstream_ref, base, timeout=120)
        if result.returncode != 0:
            raise UpdateBlocked((result.stderr or result.stdout).strip() or "git update failed")
    except Exception:
        _run(repo, "rebase", "--abort", timeout=30)
        _run(repo, "reset", "--hard", old_sha, timeout=30)
        raise

    new_sha_result = _run(repo, "rev-parse", "HEAD", check=True)
    return UpdateResult(
        changed=new_sha_result.stdout.strip() != old_sha,
        strategy=state.strategy,
        old_sha=old_sha,
        new_sha=new_sha_result.stdout.strip(),
        backup_ref=backup_ref,
    )
