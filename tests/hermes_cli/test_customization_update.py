from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli.customization_update import (
    UpdateBlocked,
    analyze_repository,
    apply_customization_update,
)


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def commit(repo: Path, name: str, content: str, message: str) -> str:
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    git(repo, "add", name)
    git(repo, "commit", "-m", message)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / "repo"
    path.mkdir()
    git(path, "init", "-b", "main")
    git(path, "config", "user.name", "Test")
    git(path, "config", "user.email", "test@example.com")
    commit(path, "shared.txt", "base\n", "base")
    git(path, "remote", "add", "upstream", "https://github.com/NousResearch/hermes-agent.git")
    git(path, "update-ref", "refs/remotes/upstream/main", "HEAD")
    return path


def advance_upstream(repo: Path, *, name: str = "upstream.txt", content: str = "upstream\n") -> str:
    original = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "--detach", "refs/remotes/upstream/main")
    sha = commit(repo, name, content, "upstream change")
    git(repo, "update-ref", "refs/remotes/upstream/main", sha)
    git(repo, "checkout", "-B", "local/customizations", original)
    return sha


def test_clean_custom_branch_selects_rebase(repo: Path) -> None:
    git(repo, "checkout", "-b", "local/customizations")
    commit(repo, "custom.txt", "custom\n", "local customization")
    advance_upstream(repo)

    state = analyze_repository(repo)

    assert state.strategy == "rebase-customizations"
    assert state.can_apply is True
    assert state.behind == 1
    assert state.custom_commit_count == 1
    assert state.custom_commits[0]["summary"] == "local customization"
    assert state.dirty is False
    assert state.backup_required is True


def test_conflict_preflight_leaves_checkout_unchanged(repo: Path) -> None:
    git(repo, "checkout", "-b", "local/customizations")
    old_sha = commit(repo, "shared.txt", "custom\n", "custom conflict")
    advance_upstream(repo, name="shared.txt", content="upstream\n")
    before_status = git(repo, "status", "--porcelain=v1")

    with pytest.raises(UpdateBlocked, match="conflict"):
        apply_customization_update(repo, fetch=False)

    assert git(repo, "rev-parse", "HEAD") == old_sha
    assert git(repo, "status", "--porcelain=v1") == before_status
    assert not (repo / ".git" / "rebase-merge").exists()


def test_dirty_worktree_blocks_apply(repo: Path) -> None:
    git(repo, "checkout", "-b", "local/customizations")
    commit(repo, "custom.txt", "custom\n", "local customization")
    advance_upstream(repo)
    (repo / "custom.txt").write_text("dirty\n")

    state = analyze_repository(repo)

    assert state.strategy == "blocked"
    assert state.can_apply is False
    assert state.dirty is True
    assert "dirty_worktree" in state.blocking_reasons


def test_fork_without_official_remote_is_blocked(repo: Path) -> None:
    git(repo, "remote", "set-url", "upstream", "https://github.com/example/fork.git")

    state = analyze_repository(repo)

    assert state.strategy == "blocked"
    assert state.can_apply is False
    assert state.upstream_ref is None
    assert "official_remote_missing" in state.blocking_reasons


def test_no_upstream_update_is_current_even_with_custom_commits(repo: Path) -> None:
    git(repo, "checkout", "-b", "local/customizations")
    commit(repo, "custom.txt", "custom\n", "local customization")

    state = analyze_repository(repo)

    assert state.strategy == "none"
    assert state.can_apply is False
    assert state.behind == 0
    assert state.custom_commit_count == 1


def test_straight_official_branch_selects_fast_forward(repo: Path) -> None:
    advance_upstream(repo)

    state = analyze_repository(repo)

    assert state.strategy == "fast-forward"
    assert state.can_apply is True
    assert state.behind == 1
    assert state.custom_commit_count == 0
    assert state.backup_required is True


def test_successful_custom_rebase_creates_backup_ref(repo: Path) -> None:
    git(repo, "checkout", "-b", "local/customizations")
    old_sha = commit(repo, "custom.txt", "custom\n", "local customization")
    upstream_sha = advance_upstream(repo)

    result = apply_customization_update(repo, fetch=False)

    assert result.changed is True
    assert result.strategy == "rebase-customizations"
    assert result.old_sha == old_sha
    assert result.backup_ref.startswith("refs/hermes-backup/")
    assert git(repo, "rev-parse", result.backup_ref) == old_sha
    assert git(repo, "merge-base", "--is-ancestor", upstream_sha, "HEAD") == ""
    assert (repo / "custom.txt").read_text() == "custom\n"
