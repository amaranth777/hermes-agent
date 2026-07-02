#!/usr/bin/env python3
"""
Memory Tool Module - Persistent Curated Memory

Provides bounded, file-backed memory that persists across sessions. Two stores:
  - MEMORY.md: agent's personal notes and observations (environment facts, project
    conventions, tool quirks, things learned)
  - USER.md: what the agent knows about the user (preferences, communication style,
    expectations, workflow habits)

Only tier1 (MEMORY.md / USER.md) is injected into the system prompt as a frozen
snapshot at session start. Mid-session tier1 writes update files on disk
immediately (durable) but do NOT change the system prompt -- this preserves the
prefix cache for the entire session. The snapshot refreshes on the next session
start.

TIERED CACHE (three tiers per target, each target = memory | user):
  - tier1 (MEMORY.md / USER.md): the ONLY tier injected into the system prompt.
    Small, high-signal, always-in-context.
  - tier2 (MEMORY.tier2.md / USER.tier2.md): overflow from tier1. Never enters
    the system prompt -- retrievable only via memory(action="search").
  - tier3 (MEMORY.tier3.md / USER.tier3.md): overflow from tier2. Same
    retrieval-only rule. Lowest priority; when tier3 itself is full, the tool
    refuses to silently drop data -- it surfaces eviction candidates and asks
    the caller to fold them into a skill (via skill_manage) or explicitly
    remove them first.
  - Overflow cascades automatically tier1 -> tier2 -> tier3 on write. Nothing
    is EVER silently deleted; when tier3 has no room the write is refused with
    the candidate entries attached so a human/model can decide what to do with
    them (see MemoryStore._cascade_demote).
  - Tiers never touch the frozen system-prompt snapshot -- adding to /
    demoting within tier2/tier3 never invalidates the prompt cache.

Entry delimiter: § (section sign). Entries can be multiline.
Character limits (not tokens) because char counts are model-independent.

Design:
- Single `memory` tool with action parameter: add, replace, remove, search
- replace/remove match by short unique substring across ALL tiers (an entry
  demoted to tier2/tier3 can still be found and edited/removed)
- search is READ-ONLY and only look at tier2/tier3 (tier1 is always already
  in context, no need to search it)
- Behavioral guidance lives in the tool schema description
- Frozen snapshot pattern: system prompt is stable, tool responses show live state
"""

import json
import logging
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from hermes_constants import get_hermes_home
from typing import Dict, Any, List, Optional

from utils import atomic_replace

# fcntl is Unix-only; on Windows use msvcrt for file locking
msvcrt = None
try:
    import fcntl
except ImportError:
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        pass

logger = logging.getLogger(__name__)

# Where memory files live — resolved dynamically so profile overrides
# (HERMES_HOME env var changes) are always respected.  The old module-level
# constant was cached at import time and could go stale if a profile switch
# happened after the first import.
def get_memory_dir() -> Path:
    """Return the profile-scoped memories directory."""
    return get_hermes_home() / "memories"

ENTRY_DELIMITER = "\n§\n"


# ---------------------------------------------------------------------------
# Memory content scanning — lightweight check for injection/exfiltration
# in content that gets injected into the system prompt.
#
# Patterns live in ``tools/threat_patterns.py`` — the single source of truth
# shared with the context-file scanner and the tool-result delimiter system.
# Memory uses the "strict" scope (broadest pattern set) because:
#  - memory entries are user-curated; the user can rewrite a flagged entry
#  - memory enters the system prompt as a FROZEN snapshot, so a poisoned
#    entry persists for the entire session and across sessions until
#    explicitly removed.
# ---------------------------------------------------------------------------

from tools.threat_patterns import first_threat_message as _first_threat_message


def _scan_memory_content(content: str) -> Optional[str]:
    """Scan memory content for injection/exfil patterns. Returns error string if blocked."""
    return _first_threat_message(content, scope="strict")


def _drift_error(path: "Path", bak_path: str) -> Dict[str, Any]:
    """Build the error dict returned when external drift is detected.

    The on-disk memory file contains content that wouldn't round-trip
    through the tool's parser/serializer — flushing would discard the
    appended/edited content from a patch tool, shell append, manual edit,
    or sister-session write. We refuse the mutation, point the operator at
    the .bak.<ts> snapshot we took, and tell them what to do next.
    """
    return {
        "success": False,
        "error": (
            f"Refusing to write {path.name}: file on disk has content that "
            f"wouldn't round-trip through the memory tool (likely added by "
            f"the patch tool, a shell append, a manual edit, or a "
            f"concurrent session). A snapshot was saved to {bak_path}. "
            f"Resolve the drift first — either rewrite the file as a clean "
            f"§-delimited list of entries, or move the extra content out — "
            f"then retry. This guard exists to prevent silent data loss "
            f"(issue #26045)."
        ),
        "drift_backup": bak_path,
        "remediation": (
            "Open the .bak file, integrate the missing entries into the "
            "memory tool one at a time via memory(action=add, content=...), "
            "then remove or rewrite the original file to a clean state."
        ),
    }


# Tiers, in cascade order. tier1 is the only one injected into the system
# prompt; tier2/tier3 are retrieval-only overflow (see module docstring).
TIERS = ("tier1", "tier2", "tier3")

# Default tier1 char limits (config keys: memory.memory_char_limit /
# memory.user_char_limit -- reused, now meaning "tier1 limit" specifically).
DEFAULT_TIER1_MEMORY_CHAR_LIMIT = 1200
DEFAULT_TIER1_USER_CHAR_LIMIT = 1200
# Default tier2/tier3 char limits (config keys: memory.tier2_char_limit /
# memory.tier3_char_limit). Same numbers apply to both memory and user
# targets, but usage is tracked independently per target.
DEFAULT_TIER2_CHAR_LIMIT = 2400
DEFAULT_TIER3_CHAR_LIMIT = 4800


class MemoryStore:
    """
    Bounded curated memory with file persistence. One instance per AIAgent.

    Three tiers per target (target = "memory" | "user"):
      - tier1: MEMORY.md / USER.md. The only tier injected into the system
        prompt (frozen snapshot, see _system_prompt_snapshot below).
      - tier2: MEMORY.tier2.md / USER.tier2.md. Retrieval-only overflow from
        tier1 -- never enters the system prompt.
      - tier3: MEMORY.tier3.md / USER.tier3.md. Retrieval-only overflow from
        tier2 -- lowest priority, last stop before a write is refused.

    Maintains two parallel states:
      - _system_prompt_snapshot: frozen at load time from tier1 ONLY, used
        for system prompt injection. Never mutated mid-session. Keeps prefix
        cache stable.
      - _entries[target][tier]: live state, mutated by tool calls, persisted
        to disk. Tool responses always reflect this live state.
        `memory_entries` / `user_entries` properties expose tier1 only, for
        backward compatibility with existing callers/tests.
    """

    # After this many failed consolidation attempts (overflow / zero-match) in
    # ONE turn, stop instructing the model to "retry in this turn" and return a
    # terminal "save skipped" result so a fragile replace/add can't loop the
    # turn to budget exhaustion and suppress the user's reply (issue #42405).
    _MAX_CONSOLIDATION_FAILURES_PER_TURN = 3

    # Max entries returned by a single search() call -- keeps a broad query
    # from dumping the entire tier2/tier3 store into one tool result.
    _SEARCH_RESULT_CAP = 20

    def __init__(
        self,
        memory_char_limit: int = DEFAULT_TIER1_MEMORY_CHAR_LIMIT,
        user_char_limit: int = DEFAULT_TIER1_USER_CHAR_LIMIT,
        tier2_char_limit: int = DEFAULT_TIER2_CHAR_LIMIT,
        tier3_char_limit: int = DEFAULT_TIER3_CHAR_LIMIT,
    ):
        # tier1 limits are per-target (kept as separate args for backward
        # compatibility with existing callers). tier2/tier3 limits are one
        # number applied to BOTH targets, but usage is tracked independently
        # per target (memory's tier2 usage never competes with user's tier2).
        self.memory_char_limit = memory_char_limit
        self.user_char_limit = user_char_limit
        self.tier2_char_limit = tier2_char_limit
        self.tier3_char_limit = tier3_char_limit

        # _entries["memory"|"user"]["tier1"|"tier2"|"tier3"] -> List[str]
        self._entries: Dict[str, Dict[str, List[str]]] = {
            "memory": {"tier1": [], "tier2": [], "tier3": []},
            "user": {"tier1": [], "tier2": [], "tier3": []},
        }
        # Frozen snapshot for system prompt -- set once at load_from_disk().
        # Built ONLY from tier1; tier2/tier3 never enter the system prompt.
        self._system_prompt_snapshot: Dict[str, str] = {"memory": "", "user": ""}
        # Per-turn counter of failed at-capacity consolidation attempts; reset
        # at each turn boundary by reset_consolidation_failures() (#42405).
        self._consolidation_failures = 0

    # -- Backward-compatible tier1 accessors --------------------------------
    # Existing code/tests read/write `store.memory_entries` / `store.user_entries`
    # expecting tier1 semantics (that's literally what MEMORY.md/USER.md are).

    @property
    def memory_entries(self) -> List[str]:
        return self._entries["memory"]["tier1"]

    @memory_entries.setter
    def memory_entries(self, value: List[str]):
        self._entries["memory"]["tier1"] = value

    @property
    def user_entries(self) -> List[str]:
        return self._entries["user"]["tier1"]

    @user_entries.setter
    def user_entries(self, value: List[str]):
        self._entries["user"]["tier1"] = value

    def reset_consolidation_failures(self) -> None:
        """Reset the per-turn consolidation-failure counter (call at turn start)."""
        self._consolidation_failures = 0

    def _consolidation_failure(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Count an at-capacity consolidation failure and degrade gracefully.

        Under the per-turn cap, return ``response`` unchanged (it already tells
        the model how to self-correct + retry in this turn). Once the cap is
        exceeded, drop the retry instruction and return a TERMINAL result so the
        model stops looping memory calls and proceeds to answer the user — a
        failed memory side effect must never block the turn's reply (#42405).
        """
        self._consolidation_failures += 1
        if self._consolidation_failures <= self._MAX_CONSOLIDATION_FAILURES_PER_TURN:
            return response
        return {
            "success": False,
            "done": True,
            "error": (
                f"Memory consolidation failed {self._consolidation_failures} times "
                "this turn. Stop retrying memory calls — leave memory unchanged for "
                "now and continue with your reply to the user. The fact can be saved "
                "in a later turn."
            ),
        }

    def load_from_disk(self):
        """Load entries from all three tiers, capture system prompt snapshot.

        The frozen snapshot is what enters the system prompt, built from
        TIER1 ONLY (tier2/tier3 are retrieval-only, they never enter the
        prompt). We scan each tier1 entry for injection/promptware patterns
        at snapshot-build time — ANY hit replaces the entry text in the
        snapshot with a placeholder like ``[BLOCKED: …]``, so a poisoned-on-disk
        memory file (supply chain, compromised tool, sister-session write)
        cannot inject into the system prompt.

        The live ``memory_entries`` / ``user_entries`` (tier1) lists keep the
        original text so the user can still see poisoned entries by
        inspecting the source files directly, and remove them — silently
        dropping them would hide the attack from the user.

        Migration: if tier1 is over its configured limit on disk (老数据 from
        before this store had tiers, or the limit was lowered), the excess is
        cascaded down into tier2/tier3 (oldest entries first) BEFORE the
        snapshot is built, so no on-disk data is ever lost to a shrinking
        limit -- it just gets re-tiered. This is logged via logger.info so a
        migration event is traceable.

        Scanning is deterministic from disk bytes, so the snapshot remains
        stable for the entire session (prefix-cache invariant holds).
        """
        mem_dir = get_memory_dir()
        mem_dir.mkdir(parents=True, exist_ok=True)

        for target in ("memory", "user"):
            for tier in TIERS:
                entries = self._read_file(self._path_for(target, tier))
                # Deduplicate within a tier (preserves order, keeps first occurrence)
                self._entries[target][tier] = list(dict.fromkeys(entries))

        # Cross-tier dedup: the same fact should not live in two tiers at
        # once (can happen if a migration ran twice, or files were hand-edited).
        # Keep the highest-priority occurrence (tier1 > tier2 > tier3).
        for target in ("memory", "user"):
            self._dedup_across_tiers(target)

        # Migration: pull any tier1 overflow (vs the CURRENT configured
        # limit) down into tier2/tier3 before building the snapshot.
        for target in ("memory", "user"):
            self._migrate_tier1_overflow(target)

        # Persist any changes the dedup/migration steps made (no-op if nothing moved).
        for target in ("memory", "user"):
            for tier in TIERS:
                self.save_to_disk(target, tier)

        # Sanitize tier1 entries for the system-prompt snapshot only. Live
        # state (memory_entries / user_entries) keeps the raw text so the
        # user can see + remove poisoned entries via the memory tool.
        sanitized_memory = self._sanitize_entries_for_snapshot(self.memory_entries, "MEMORY.md")
        sanitized_user = self._sanitize_entries_for_snapshot(self.user_entries, "USER.md")

        # Capture frozen snapshot for system prompt injection (tier1 ONLY).
        self._system_prompt_snapshot = {
            "memory": self._render_block("memory", sanitized_memory),
            "user": self._render_block("user", sanitized_user),
        }

    def _dedup_across_tiers(self, target: str) -> None:
        """Remove an entry from lower tiers if it already exists in a higher one.

        Priority order is tier1 > tier2 > tier3 (tier1 is truth; a lower-tier
        duplicate is redundant and only wastes budget).
        """
        seen: set = set()
        for tier in TIERS:  # tier1, tier2, tier3 in that order
            entries = self._entries[target][tier]
            deduped = []
            for e in entries:
                if e in seen:
                    continue
                seen.add(e)
                deduped.append(e)
            self._entries[target][tier] = deduped

    def _tier_limit(self, target: str, tier: str) -> int:
        if tier == "tier1":
            return self.user_char_limit if target == "user" else self.memory_char_limit
        if tier == "tier2":
            return self.tier2_char_limit
        return self.tier3_char_limit

    def _migrate_tier1_overflow(self, target: str) -> None:
        """If tier1 on-disk content exceeds its current limit, cascade the
        oldest entries down into tier2 (and tier3 if needed) so nothing is
        lost when the configured limit shrinks or legacy data predates tiering.
        """
        tier1 = self._entries[target]["tier1"]
        limit = self._tier_limit(target, "tier1")
        if not tier1 or len(ENTRY_DELIMITER.join(tier1)) <= limit:
            return

        moved = 0
        while tier1 and len(ENTRY_DELIMITER.join(tier1)) > limit:
            oldest = tier1[0]
            ok = self._demote_one(target, "tier2", oldest)
            if not ok:
                # tier2 (and tier3 beneath it) truly has no room. Leave the
                # remaining tier1 entries in place rather than lose data --
                # this is a startup migration, not a user-initiated write, so
                # there's no one to hand an actionable error to right now.
                # Log loudly; the oversized tier1 will surface on the next
                # add()/replace() call via the normal cascade/refusal path.
                logger.warning(
                    "Memory migration for %s: tier1 still over limit (%d/%d chars) "
                    "but tier2/tier3 have no room to absorb more. %d entries moved so far.",
                    target, len(ENTRY_DELIMITER.join(tier1)), limit, moved,
                )
                break
            tier1.pop(0)
            moved += 1

        if moved:
            logger.info(
                "Memory migration: moved %d oldest entr%s from %s tier1 to lower tiers "
                "(tier1 limit=%d chars).",
                moved, "y" if moved == 1 else "ies", target, limit,
            )

    @staticmethod
    def _sanitize_entries_for_snapshot(entries: List[str], filename: str) -> List[str]:
        """Return ``entries`` with any threat-matching entry replaced by a placeholder.

        Each entry is scanned with the shared threat-pattern library at the
        ``"strict"`` scope (same as memory writes).  On match, the entry is
        replaced in the returned list with ``"[BLOCKED: <filename> entry
        contained threat pattern: <ids>. Removed from system prompt.]"`` —
        the placeholder enters the snapshot, the original entry stays in
        live state for the user to inspect and delete.

        Empty or already-block-marker entries pass through unchanged.
        """
        from tools.threat_patterns import scan_for_threats

        sanitized: List[str] = []
        for entry in entries:
            if not entry or entry.startswith("[BLOCKED:"):
                sanitized.append(entry)
                continue
            findings = scan_for_threats(entry, scope="strict")
            if findings:
                logger.warning(
                    "Memory entry from %s blocked at load time: %s",
                    filename, ", ".join(findings),
                )
                sanitized.append(
                    f"[BLOCKED: {filename} entry contained threat pattern(s): "
                    f"{', '.join(findings)}. Removed from system prompt; "
                    f"use memory(action=remove) "
                    f"to delete the original.]"
                )
            else:
                sanitized.append(entry)
        return sanitized

    @staticmethod
    @contextmanager
    def _file_lock(path: Path):
        """Acquire an exclusive file lock for read-modify-write safety.

        Uses a separate .lock file so the memory file itself can still be
        atomically replaced via os.replace().
        """
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        if fcntl is None and msvcrt is None:
            yield
            return

        fd = open(lock_path, "a+", encoding="utf-8")
        try:
            if fcntl:
                fcntl.flock(fd, fcntl.LOCK_EX)
            else:
                fd.seek(0)
                msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
            yield
        finally:
            if fcntl:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except (OSError, IOError):
                    pass
            elif msvcrt:
                try:
                    fd.seek(0)
                    msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
                except (OSError, IOError):
                    pass
            fd.close()

    @staticmethod
    def _path_for(target: str, tier: str = "tier1") -> Path:
        mem_dir = get_memory_dir()
        base = "USER" if target == "user" else "MEMORY"
        if tier == "tier1":
            return mem_dir / f"{base}.md"
        return mem_dir / f"{base}.{tier}.md"

    def _reload_target(self, target: str, *, skip_drift: bool = False, tier: str = "tier1") -> Optional[str]:
        """Re-read entries from disk into in-memory state for one tier.

        Called under file lock to get the latest state before mutating.
        Returns the backup path if external drift was detected (the on-disk
        file contains content that wouldn't round-trip through our
        parser/serializer, OR an entry larger than the tier's char limit).
        When drift is detected the caller must abort the mutation —
        flushing would discard the un-roundtrippable content.
        Returns None on clean reload.

        When *skip_drift* is True the round-trip / entry-size check is
        bypassed.  Used by the ``add`` action which appends without
        rewriting, so existing content is never clobbered.
        """
        path = self._path_for(target, tier)
        bak = None if skip_drift else self._detect_external_drift(target, tier)
        fresh = self._read_file(path)
        fresh = list(dict.fromkeys(fresh))  # deduplicate
        self._entries[target][tier] = fresh
        return bak

    def _reload_all_tiers(self, target: str, *, skip_drift: bool = False) -> Optional[str]:
        """Re-read all three tiers for a target. Returns first drift backup found, if any."""
        for tier in TIERS:
            bak = self._reload_target(target, skip_drift=skip_drift, tier=tier)
            if bak:
                return bak
        return None

    def save_to_disk(self, target: str, tier: str = "tier1"):
        """Persist one tier's entries to its file. Called after every mutation."""
        get_memory_dir().mkdir(parents=True, exist_ok=True)
        self._write_file(self._path_for(target, tier), self._entries[target][tier])

    def _entries_for(self, target: str, tier: str = "tier1") -> List[str]:
        return self._entries[target][tier]

    def _set_entries(self, target: str, entries: List[str], tier: str = "tier1"):
        self._entries[target][tier] = entries

    def _char_count(self, target: str, tier: str = "tier1") -> int:
        entries = self._entries_for(target, tier)
        if not entries:
            return 0
        return len(ENTRY_DELIMITER.join(entries))

    def _char_limit(self, target: str, tier: str = "tier1") -> int:
        return self._tier_limit(target, tier)

    def _find_entry(self, target: str, old_text: str) -> Optional[Dict[str, Any]]:
        """Search old_text as a substring across tier1 -> tier2 -> tier3.

        Returns None if no tier has any match. Otherwise returns a dict:
          {"tier": <first tier with a match>, "matches": [(idx, entry), ...]}
        Search stops at the FIRST tier with any match (matching the existing
        single-tier semantics: ambiguity is only checked within that tier).
        """
        for tier in TIERS:
            entries = self._entries_for(target, tier)
            matches = [(i, e) for i, e in enumerate(entries) if old_text in e]
            if matches:
                return {"tier": tier, "matches": matches}
        return None

    def _all_entries_flat(self, target: str) -> List[str]:
        """All entries across all three tiers, for cross-tier duplicate checks."""
        flat: List[str] = []
        for tier in TIERS:
            flat.extend(self._entries_for(target, tier))
        return flat

    def _demote_one(self, target: str, dest_tier: str, entry: str) -> bool:
        """Try to place `entry` into `dest_tier` for `target`, cascading further
        down (tier2 -> tier3) if `dest_tier` itself has no room.

        Mutates in-memory state (does NOT write to disk -- caller commits).
        Returns True if the entry was placed somewhere (dest_tier or deeper),
        False if there was no room anywhere in the cascade (caller must not
        proceed with the demotion that triggered this call).
        """
        dest_entries = self._entries_for(target, dest_tier)
        limit = self._tier_limit(target, dest_tier)
        candidate_total = len(ENTRY_DELIMITER.join(dest_entries + [entry]))

        if candidate_total <= limit:
            dest_entries.append(entry)
            return True

        if dest_tier == "tier3":
            # Bottom of the cascade -- no further tier to push into.
            return False

        # dest_tier has no room -- try to make room by pushing ITS oldest
        # entry one tier further down before giving up.
        next_tier = "tier3" if dest_tier == "tier2" else None
        if next_tier is None:
            return False
        if not dest_entries:
            # dest_tier is empty but the single entry itself is too big for
            # dest_tier's limit -- pushing further down won't help either
            # unless the deeper tier's limit is bigger, so just try it directly.
            return self._demote_one(target, next_tier, entry)

        oldest = dest_entries[0]
        if not self._demote_one(target, next_tier, oldest):
            return False
        dest_entries.pop(0)
        # Retry placing the original entry into dest_tier now that it has room.
        return self._demote_one(target, dest_tier, entry)

    def add(self, target: str, content: str) -> Dict[str, Any]:
        """Append a new entry to tier1, cascading overflow down to tier2/tier3.

        Cascade algorithm (all validated before ANY file is written):
          1. Empty / injection checks (unchanged).
          2. Cross-tier duplicate check -- if content already exists anywhere
             (tier1/2/3), no-op success (unchanged semantics, wider scope).
          3. If content alone exceeds the tier1 limit, reject outright (it can
             never fit even after demoting everything else out of tier1).
          4. If content fits in tier1 as-is, fast path: append + save, done.
          5. Otherwise, demote tier1's oldest entries (FIFO) one at a time into
             tier2 (cascading into tier3 if tier2 itself is full) until the new
             content fits in tier1. If at any point tier3 has no room to accept
             what's being pushed into it, THE WHOLE OPERATION ABORTS: no file is
             written, and the error names the tier3 eviction candidates so the
             caller can fold them into a skill or explicitly delete first.
        """
        content = content.strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}

        # Scan for injection/exfiltration before accepting
        scan_error = _scan_memory_content(content)
        if scan_error:
            return {"success": False, "error": scan_error}

        with self._file_lock(self._path_for(target)):
            # Re-read from disk under lock to pick up writes from other sessions.
            # For add (append-only), we skip the drift guard — appending never
            # clobbers existing content, so round-trip mismatches from prior
            # tool-written entries in the same session are harmless.  The drift
            # guard remains active for replace/remove where full-file rewrite
            # would discard un-roundtrippable content (issue #26045).
            self._reload_all_tiers(target, skip_drift=True)

            # Cross-tier duplicate check.
            if content in self._all_entries_flat(target):
                return self._success_response(target, "Entry already exists (no duplicate added).")

            tier1_limit = self._tier_limit(target, "tier1")
            if len(content) > tier1_limit:
                return {
                    "success": False,
                    "error": (
                        f"This entry alone is {len(content):,} chars, which exceeds the tier1 "
                        f"limit of {tier1_limit:,} chars. Shorten it before saving — no amount "
                        f"of cascading frees enough room for a single oversized entry."
                    ),
                }

            tier1 = self._entries_for(target, "tier1")
            new_total = len(ENTRY_DELIMITER.join(tier1 + [content]))

            if new_total <= tier1_limit:
                # Fast path: fits as-is, behavior identical to the pre-tiering add().
                tier1.append(content)
                self.save_to_disk(target, "tier1")
                return self._success_response(target, "Entry added.")

            # Need to cascade: work on snapshots so a failed cascade writes nothing.
            working_tier1 = list(tier1)
            snapshot_tier2 = list(self._entries_for(target, "tier2"))
            snapshot_tier3 = list(self._entries_for(target, "tier3"))

            while working_tier1 and len(ENTRY_DELIMITER.join(working_tier1 + [content])) > tier1_limit:
                oldest = working_tier1[0]
                if not self._demote_one(target, "tier2", oldest):
                    # Cascade failed -- tier3 has no room either. Roll back the
                    # in-memory tier2/tier3 state to the pre-attempt snapshot
                    # (._demote_one may have partially mutated them) and abort
                    # without writing ANY file.
                    self._entries[target]["tier2"] = snapshot_tier2
                    self._entries[target]["tier3"] = snapshot_tier3
                    return self._consolidation_failure(self._tier3_full_error(target))
                working_tier1.pop(0)

            # Cascade validated -- commit tier1 + whichever lower tiers changed.
            working_tier1.append(content)
            self._entries[target]["tier1"] = working_tier1
            self.save_to_disk(target, "tier1")
            self.save_to_disk(target, "tier2")
            self.save_to_disk(target, "tier3")

        return self._success_response(target, "Entry added (older entries cascaded to tier2/tier3).")

    def _tier3_full_error(self, target: str) -> Dict[str, Any]:
        """Build the refusal returned when a cascade can't free room even in tier3.

        Surfaces tier3's oldest entries as eviction candidates -- the tool
        NEVER deletes them itself. The caller must fold them into a skill
        (skill_manage) or explicitly memory(action="remove") them, then retry.
        """
        tier3 = self._entries_for(target, "tier3")
        limit = self._tier_limit(target, "tier3")
        current = self._char_count(target, "tier3")
        candidates = self._previews(tier3[:5])
        return {
            "success": False,
            "error": (
                f"tier3 for '{target}' is full ({current:,}/{limit:,} chars) and has no room "
                f"to absorb more cascaded entries. Nothing was written (tier1/tier2/tier3 all "
                f"unchanged). Before retrying, either: (a) fold one or more of the tier3 "
                f"eviction_candidates below into an existing skill via skill_manage, then "
                f"memory(action='remove') the folded entry to free room, or (b) if a candidate "
                f"is simply stale, remove it directly. Then retry the add."
            ),
            "eviction_candidates": candidates,
            "usage": f"{current:,}/{limit:,}",
        }

    def replace(self, target: str, old_text: str, new_content: str) -> Dict[str, Any]:
        """Find entry containing old_text substring across tier1->tier2->tier3, replace it.

        Search stops at the first tier with any match (mirrors add's tiering:
        an entry demoted to tier2/tier3 can still be found and edited). The
        char-budget check is against THAT tier's own limit -- editing a tier3
        entry never affects tier1's budget.
        """
        old_text = old_text.strip()
        new_content = new_content.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        if not new_content:
            return {"success": False, "error": "new_content cannot be empty. Use 'remove' to delete entries."}

        # Scan replacement content for injection/exfiltration
        scan_error = _scan_memory_content(new_content)
        if scan_error:
            return {"success": False, "error": scan_error}

        with self._file_lock(self._path_for(target)):
            bak = self._reload_all_tiers(target)
            if bak:
                return _drift_error(self._path_for(target), bak)

            found = self._find_entry(target, old_text)

            if not found:
                return self._consolidation_failure({
                    "success": False,
                    "error": f"No entry matched '{old_text}'. Check current_entries below and retry with the exact text of the entry you want to replace.",
                    "current_entries": self._all_entries_flat(target),
                })

            tier = found["tier"]
            matches = found["matches"]
            entries = self._entries_for(target, tier)

            if len(matches) > 1:
                # If all matches are identical (exact duplicates), operate on the first one
                unique_texts = {e for _, e in matches}
                if len(unique_texts) > 1:
                    previews = self._previews([e for _, e in matches])
                    return {
                        "success": False,
                        "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                        "matches": previews,
                    }
                # All identical -- safe to replace just the first

            idx = matches[0][0]
            limit = self._char_limit(target, tier)

            # Check that replacement doesn't blow THIS TIER's own budget
            test_entries = entries.copy()
            test_entries[idx] = new_content
            new_total = len(ENTRY_DELIMITER.join(test_entries))

            if new_total > limit:
                current = self._char_count(target, tier)
                return self._consolidation_failure({
                    "success": False,
                    "error": (
                        f"Replacement would put {tier} at {new_total:,}/{limit:,} chars. "
                        f"Shorten the new content, or 'remove' other stale or less important "
                        f"entries to make room (see current_entries below), then retry — all "
                        f"in this turn."
                    ),
                    "current_entries": entries,
                    "usage": f"{current:,}/{limit:,}",
                })

            entries[idx] = new_content
            self._set_entries(target, entries, tier)
            self.save_to_disk(target, tier)

        return self._success_response(target, "Entry replaced.")

    def remove(self, target: str, old_text: str) -> Dict[str, Any]:
        """Remove the entry containing old_text substring, searched across all tiers."""
        old_text = old_text.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}

        with self._file_lock(self._path_for(target)):
            bak = self._reload_all_tiers(target)
            if bak:
                return _drift_error(self._path_for(target), bak)

            found = self._find_entry(target, old_text)

            if not found:
                return self._consolidation_failure({
                    "success": False,
                    "error": f"No entry matched '{old_text}'. Check current_entries below and retry with the exact text of the entry you want to remove.",
                    "current_entries": self._all_entries_flat(target),
                })

            tier = found["tier"]
            matches = found["matches"]
            entries = self._entries_for(target, tier)

            if len(matches) > 1:
                # If all matches are identical (exact duplicates), remove the first one
                unique_texts = {e for _, e in matches}
                if len(unique_texts) > 1:
                    previews = self._previews([e for _, e in matches])
                    return {
                        "success": False,
                        "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                        "matches": previews,
                    }
                # All identical -- safe to remove just the first

            idx = matches[0][0]
            entries.pop(idx)
            self._set_entries(target, entries, tier)
            self.save_to_disk(target, tier)

        return self._success_response(target, "Entry removed.")

    def search(self, target: str, query: str, tier: str = "all") -> Dict[str, Any]:
        """Read-only substring search over tier2/tier3 (never tier1 -- it's always in context).

        `tier` is "tier2", "tier3", or "all" (default). Case-insensitive.
        Results are capped at _SEARCH_RESULT_CAP to avoid flooding the tool
        result on a broad query; `total_matches` reports the true count even
        when truncated.
        """
        query = (query or "").strip()
        if not query:
            return {"success": False, "error": "query cannot be empty."}
        if tier not in {"tier2", "tier3", "all"}:
            return {"success": False, "error": "tier must be 'tier2', 'tier3', or 'all'."}

        search_tiers = ["tier2", "tier3"] if tier == "all" else [tier]
        needle = query.lower()

        hits: List[Dict[str, str]] = []
        for t in search_tiers:
            for entry in self._entries_for(target, t):
                if needle in entry.lower():
                    hits.append({"tier": t, "entry": entry})

        total = len(hits)
        truncated = hits[: self._SEARCH_RESULT_CAP]
        return {
            "success": True,
            "target": target,
            "query": query,
            "total_matches": total,
            "returned": len(truncated),
            "results": truncated,
        }

    def apply_batch(self, target: str, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply a sequence of add/replace/remove ops to one target atomically.

        All operations are validated and applied against the FINAL budget --
        intermediate overflow is irrelevant. This lets the model free space
        (remove/replace) and add new entries in a SINGLE tool call instead of
        the multi-turn consolidate-then-retry dance that re-sends the whole
        conversation context several times.

        Semantics: all-or-nothing. If any op is malformed, doesn't match, or
        the net result would exceed the char limit, NOTHING is written and an
        error is returned describing the first failure plus the live state.
        """
        if not operations:
            return {"success": False, "error": "operations list is empty."}

        # Scan every add/replace content for injection/exfil BEFORE touching
        # disk -- a single poisoned op rejects the whole batch.
        for i, op in enumerate(operations):
            act = (op or {}).get("action")
            new_content = (op or {}).get("content")
            if act in {"add", "replace"} and new_content:
                scan_error = _scan_memory_content(new_content)
                if scan_error:
                    return {"success": False, "error": f"Operation {i + 1}: {scan_error}"}

        with self._file_lock(self._path_for(target)):
            bak = self._reload_target(target)
            if bak:
                return _drift_error(self._path_for(target), bak)

            # Work on a copy; only commit if the whole batch validates.
            working: List[str] = list(self._entries_for(target))
            limit = self._char_limit(target)

            for i, op in enumerate(operations):
                op = op or {}
                act = op.get("action")
                content = (op.get("content") or "").strip()
                old_text = (op.get("old_text") or "").strip()
                pos = f"Operation {i + 1} ({act or 'unknown'})"

                if act == "add":
                    if not content:
                        return self._batch_error(target, f"{pos}: content is required.")
                    if content in working:
                        continue  # idempotent -- skip duplicate, don't fail the batch
                    working.append(content)

                elif act == "replace":
                    if not old_text:
                        return self._batch_error(target, f"{pos}: old_text is required.")
                    if not content:
                        return self._batch_error(
                            target,
                            f"{pos}: content is required (use action='remove' to delete).",
                        )
                    matches = [j for j, e in enumerate(working) if old_text in e]
                    if not matches:
                        return self._batch_error(target, f"{pos}: no entry matched '{old_text}'.")
                    if len({working[j] for j in matches}) > 1:
                        return self._batch_error(
                            target,
                            f"{pos}: '{old_text}' matched multiple distinct entries -- be more specific.",
                        )
                    working[matches[0]] = content

                elif act == "remove":
                    if not old_text:
                        return self._batch_error(target, f"{pos}: old_text is required.")
                    matches = [j for j, e in enumerate(working) if old_text in e]
                    if not matches:
                        return self._batch_error(target, f"{pos}: no entry matched '{old_text}'.")
                    if len({working[j] for j in matches}) > 1:
                        return self._batch_error(
                            target,
                            f"{pos}: '{old_text}' matched multiple distinct entries -- be more specific.",
                        )
                    working.pop(matches[0])

                else:
                    return self._batch_error(
                        target,
                        f"{pos}: unknown action. Use add, replace, or remove.",
                    )

            # Budget check against the FINAL state only.
            new_total = len(ENTRY_DELIMITER.join(working)) if working else 0
            if new_total > limit:
                current = self._char_count(target)
                return self._consolidation_failure({
                    "success": False,
                    "error": (
                        f"After applying all {len(operations)} operations, memory would be at "
                        f"{new_total:,}/{limit:,} chars -- over the limit. Remove or shorten more "
                        f"entries in the same batch (see current_entries below), then retry."
                    ),
                    "current_entries": self._entries_for(target),
                    "usage": f"{current:,}/{limit:,}",
                })

            # Commit.
            self._set_entries(target, working)
            self.save_to_disk(target)

        return self._success_response(target, f"Applied {len(operations)} operation(s).")

    def _batch_error(self, target: str, message: str) -> Dict[str, Any]:
        """Build a batch-abort error that reports live (uncommitted) state."""
        current = self._char_count(target)
        limit = self._char_limit(target)
        return self._consolidation_failure({
            "success": False,
            "error": message + " No operations were applied (batch is all-or-nothing).",
            "current_entries": self._entries_for(target),
            "usage": f"{current:,}/{limit:,}",
        })

    def format_for_system_prompt(self, target: str) -> Optional[str]:
        """
        Return the frozen snapshot for system prompt injection.

        This returns the state captured at load_from_disk() time, NOT the live
        state. Mid-session writes do not affect this. This keeps the system
        prompt stable across all turns, preserving the prefix cache.

        Returns None if the snapshot is empty (no entries at load time).
        """
        block = self._system_prompt_snapshot.get(target, "")
        return block if block else None

    # -- Internal helpers --

    @staticmethod
    def _previews(entries: List[str], width: int = 80) -> List[str]:
        """Truncated one-line previews of entries for error feedback."""
        return [e[:width] + ("..." if len(e) > width else "") for e in entries]

    def _success_response(self, target: str, message: str = None) -> Dict[str, Any]:
        # A successful write means the consolidation loop made progress, so the
        # per-turn failure budget resets (the cap counts consecutive failures,
        # not lifetime ones within a turn) (#42405).
        self._consolidation_failures = 0
        entries = self._entries_for(target)
        current = self._char_count(target)
        limit = self._char_limit(target)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0

        # The success response is intentionally TERMINAL: it confirms the write
        # landed and tells the model to stop. We do NOT echo the full entries
        # list here -- dumping it invites the model to "find more to fix" and
        # re-issue the same operations (observed thrash: the correct batch on
        # call 1, then 5 redundant repeats). Entries are only shown on the
        # error/over-budget paths, where the model genuinely needs them to
        # decide what to consolidate.
        resp = {
            "success": True,
            "done": True,
            "target": target,
            "usage": f"{pct}% — {current:,}/{limit:,} chars",
            "entry_count": len(entries),
        }
        if message:
            resp["message"] = message
        resp["note"] = "Write saved. This update is complete — do not repeat it."
        return resp

    def _render_block(self, target: str, entries: List[str]) -> str:
        """Render a system prompt block with header and usage indicator."""
        if not entries:
            return ""

        limit = self._char_limit(target)
        content = ENTRY_DELIMITER.join(entries)
        current = len(content)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0

        if target == "user":
            header = f"USER PROFILE (who the user is) [{pct}% — {current:,}/{limit:,} chars]"
        else:
            header = f"MEMORY (your personal notes) [{pct}% — {current:,}/{limit:,} chars]"

        separator = "═" * 46
        return f"{separator}\n{header}\n{separator}\n{content}"

    @staticmethod
    def _read_file(path: Path) -> List[str]:
        """Read a memory file and split into entries.

        No file locking needed: _write_file uses atomic rename, so readers
        always see either the previous complete file or the new complete file.
        """
        if not path.exists():
            return []
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, IOError):
            return []

        if not raw.strip():
            return []

        # Use ENTRY_DELIMITER for consistency with _write_file. Splitting by "§"
        # alone would incorrectly split entries that contain "§" in their content.
        entries = [e.strip() for e in raw.split(ENTRY_DELIMITER)]
        return [e for e in entries if e]

    def _detect_external_drift(self, target: str, tier: str = "tier1") -> Optional[str]:
        """Return a backup-path string if on-disk content shows external drift.

        The memory file is supposed to be a list of small entries the tool
        wrote, joined by §. Detect drift via two signals:

        1. Round-trip mismatch — re-parsing and re-serializing the file
           doesn't produce identical bytes (rare; would catch oddly-encoded
           delimiters).
        2. Entry-size overflow — any single parsed entry exceeds this tier's
           char limit. The tool budgets each tier's ENTIRE file against that
           tier's own limit; no single tool-written entry can exceed it.
           When we see one entry larger than the limit, an external writer
           (patch tool, shell append, manual edit, sister session) appended
           free-form content into what the tool will treat as one entry.
           Flushing would then truncate that entry to the model's new
           content, discarding the appended bytes — issue #26045.

        Returns the absolute path of the .bak file when drift was found and
        backed up; returns None when the file looks tool-shaped.

        Note: this is an INSTANCE method (not static) because we need the
        per-target/per-tier char_limit for signal #2.
        """
        path = self._path_for(target, tier)
        if not path.exists():
            return None
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, IOError):
            return None
        if not raw.strip():
            return None

        parsed = [e.strip() for e in raw.split(ENTRY_DELIMITER) if e.strip()]
        roundtrip = ENTRY_DELIMITER.join(parsed)

        char_limit = self._char_limit(target, tier)
        max_entry_len = max((len(e) for e in parsed), default=0)

        drift_detected = (raw.strip() != roundtrip) or (max_entry_len > char_limit)
        if not drift_detected:
            return None

        # Drift confirmed — snapshot the file so the operator can recover
        # whatever the external writer added, then return the .bak path so
        # the caller can refuse the mutation.
        ts = int(time.time())
        bak_path = path.with_suffix(path.suffix + f".bak.{ts}")
        try:
            bak_path.write_text(raw, encoding="utf-8")
        except (OSError, IOError):
            return str(bak_path) + " (BACKUP FAILED — file unchanged on disk)"
        return str(bak_path)

    @staticmethod
    def _write_file(path: Path, entries: List[str]):
        """Write entries to a memory file using atomic temp-file + rename.

        Previous implementation used open("w") + flock, but "w" truncates the
        file *before* the lock is acquired, creating a race window where
        concurrent readers see an empty file. Atomic rename avoids this:
        readers always see either the old complete file or the new one.
        """
        content = ENTRY_DELIMITER.join(entries) if entries else ""
        try:
            # Write to temp file in same directory (same filesystem for atomic rename)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(path.parent), suffix=".tmp", prefix=".mem_"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                atomic_replace(tmp_path, path)
            except BaseException:
                # Clean up temp file on any failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except (OSError, IOError) as e:
            raise RuntimeError(f"Failed to write memory file {path}: {e}")


def load_on_disk_store() -> "MemoryStore":
    """Build a fresh on-disk :class:`MemoryStore`, honoring configured char limits.

    Use this from any context that has no live agent (the messaging gateway, the
    Desktop GUI, the bare CLI ``/memory`` handler) but still needs to read or
    apply approved memory writes. Mirrors how the live agent constructs its store
    in ``agent/agent_init.py`` — including the user's ``memory.memory_char_limit``
    / ``memory.user_char_limit`` (tier1) and ``memory.tier2_char_limit`` /
    ``memory.tier3_char_limit`` overrides — so an approval applied without a
    live agent enforces the SAME caps as one applied with one.

    Falls back to the built-in defaults if config can't be loaded, so this can
    never raise on a missing/unreadable config.
    """
    memory_char_limit = DEFAULT_TIER1_MEMORY_CHAR_LIMIT
    user_char_limit = DEFAULT_TIER1_USER_CHAR_LIMIT
    tier2_char_limit = DEFAULT_TIER2_CHAR_LIMIT
    tier3_char_limit = DEFAULT_TIER3_CHAR_LIMIT
    try:
        from hermes_cli.config import load_config

        mem_cfg = (load_config() or {}).get("memory", {}) or {}
        memory_char_limit = int(mem_cfg.get("memory_char_limit", memory_char_limit))
        user_char_limit = int(mem_cfg.get("user_char_limit", user_char_limit))
        tier2_char_limit = int(mem_cfg.get("tier2_char_limit", tier2_char_limit))
        tier3_char_limit = int(mem_cfg.get("tier3_char_limit", tier3_char_limit))
    except Exception:
        pass  # config optional — fall back to defaults rather than break /memory

    store = MemoryStore(
        memory_char_limit=memory_char_limit,
        user_char_limit=user_char_limit,
        tier2_char_limit=tier2_char_limit,
        tier3_char_limit=tier3_char_limit,
    )
    store.load_from_disk()
    return store


def _apply_write_gate(action: str, target: str, content: Optional[str],
                      old_text: Optional[str]) -> Optional[str]:
    """Evaluate the memory write gate. Returns a JSON tool-result string when
    the write should NOT proceed normally (blocked or staged), or None when the
    caller should perform the real write.

    Only the mutating actions (add/replace/remove) are gated.
    """
    if action not in {"add", "replace", "remove"}:
        return None

    try:
        from tools import write_approval as wa
    except Exception:
        # If the gate module can't load, fail open (current behaviour) rather
        # than blocking all memory writes.
        return None

    # Build a small inline summary/detail for the foreground approval prompt.
    label = "user profile" if target == "user" else "memory"
    if action == "add":
        summary = f"add to {label}"
        detail = content or ""
    elif action == "replace":
        summary = f"replace in {label}"
        detail = f"old: {old_text}\nnew: {content}"
    else:  # remove
        summary = f"remove from {label}"
        detail = old_text or ""

    decision = wa.evaluate_gate(wa.MEMORY, inline_summary=summary, inline_detail=detail)

    if decision.allow:
        return None

    if decision.blocked:
        return tool_error(decision.message, success=False)

    # stage
    payload = {
        "action": action,
        "target": target,
        "content": content,
        "old_text": old_text,
    }
    record = wa.stage_write(
        wa.MEMORY, payload,
        summary=f"{summary}: {detail[:120]}",
        origin=wa.current_origin(),
    )
    return json.dumps(
        {"success": True, "staged": True, "pending_id": record["id"],
         "message": decision.message},
        ensure_ascii=False,
    )


def _apply_batch_write_gate(target: str, operations: List[Dict[str, Any]]) -> Optional[str]:
    """Evaluate the write gate for a batch of memory operations.

    Returns a JSON tool-result string when the batch should NOT proceed
    (blocked or staged), or None when the caller should perform the real
    batch write. The whole batch is gated as a single unit.
    """
    try:
        from tools import write_approval as wa
    except Exception:
        return None

    label = "user profile" if target == "user" else "memory"
    summary = f"apply {len(operations)} op(s) to {label}"
    detail_lines = []
    for op in operations:
        op = op or {}
        act = op.get("action", "?")
        if act == "remove":
            detail_lines.append(f"- remove: {op.get('old_text', '')}")
        elif act == "replace":
            detail_lines.append(f"- replace: {op.get('old_text', '')} -> {op.get('content', '')}")
        else:
            detail_lines.append(f"- {act}: {op.get('content', '')}")
    detail = "\n".join(detail_lines)

    decision = wa.evaluate_gate(wa.MEMORY, inline_summary=summary, inline_detail=detail)

    if decision.allow:
        return None

    if decision.blocked:
        return tool_error(decision.message, success=False)

    payload = {"action": "batch", "target": target, "operations": operations}
    record = wa.stage_write(
        wa.MEMORY, payload,
        summary=f"{summary}: {detail[:120]}",
        origin=wa.current_origin(),
    )
    return json.dumps(
        {"success": True, "staged": True, "pending_id": record["id"],
         "message": decision.message},
        ensure_ascii=False,
    )


def _missing_old_text_error(store: "MemoryStore", target: str, action: str) -> str:
    """Build a recoverable error for a replace/remove call that arrived without
    ``old_text``.

    ``replace``/``remove`` are inherently targeted -- without ``old_text`` there
    is no entry to act on, so we cannot fulfil the call. But returning a bare
    "old_text is required" is a dead-end: some structured-output clients omit the
    optional ``old_text`` field (it isn't, and can't be, schema-required without
    a top-level combinator the Codex backend rejects -- see
    tests/tools/test_memory_tool_schema.py). So instead we return the current
    entry inventory plus an explicit retry instruction, letting the model reissue
    the call with ``old_text`` set to a unique substring of the entry it means.
    Mirrors the batch path's ``_batch_error`` shape. (issues #43412, #49466)
    """
    entries = store._entries_for(target)
    current = store._char_count(target)
    limit = store._char_limit(target)
    return json.dumps(
        {
            "success": False,
            "error": (
                f"'{action}' needs old_text -- a short unique substring of the entry "
                f"to {action}. None was provided. Reissue the {action} with old_text "
                f"set to part of one of the current_entries below."
            ),
            "current_entries": entries,
            "usage": f"{current:,}/{limit:,}",
        },
        ensure_ascii=False,
    )


def memory_tool(
    action: str = None,
    target: str = "memory",
    content: str = None,
    old_text: str = None,
    operations: Optional[List[Dict[str, Any]]] = None,
    query: str = None,
    tier: str = "all",
    store: Optional[MemoryStore] = None,
) -> str:
    """
    Single entry point for the memory tool. Dispatches to MemoryStore methods.

    Three shapes:
      - Single op: action + (content / old_text).
      - Batch:     operations=[{action, content?, old_text?}, ...] applied
                   atomically against the final char budget in ONE call.
      - Search:    action="search" + query (+ optional tier). READ-ONLY,
                   bypasses the write-approval gate entirely (nothing is
                   mutated), only looks at tier2/tier3 (tier1 is already in
                   context, no need to search it).

    Returns JSON string with results.
    """
    if store is None:
        return tool_error("Memory is not available. It may be disabled in config or this environment.", success=False)

    if target not in {"memory", "user"}:
        return tool_error(f"Invalid target '{target}'. Use 'memory' or 'user'.", success=False)

    # --- Search path (read-only, no gate, no batch support) ---------------
    if action == "search":
        if not query:
            return tool_error("query is required for 'search' action.", success=False)
        result = store.search(target, query, tier or "all")
        return json.dumps(result, ensure_ascii=False)

    # --- Batch path -------------------------------------------------------
    if operations:
        if not isinstance(operations, list):
            return tool_error("operations must be a list of {action, content?, old_text?} objects.", success=False)
        gate_result = _apply_batch_write_gate(target, operations)
        if gate_result is not None:
            return gate_result
        result = store.apply_batch(target, operations)
        return json.dumps(result, ensure_ascii=False)

    # --- Single-op path ---------------------------------------------------
    # Validate required params BEFORE the gate so an invalid write is rejected
    # immediately instead of being staged and only failing at approve time.
    if action == "add" and not content:
        return tool_error("Content is required for 'add' action.", success=False)
    if action == "replace" and (not old_text or not content):
        missing = "old_text" if not old_text else "content"
        if not old_text:
            # The client/model omitted old_text. Replace is inherently targeted
            # -- we can't guess which entry. Return the current inventory plus a
            # retry instruction so the model can reissue with old_text set,
            # instead of hitting a dead-end error. (issues #43412, #49466)
            return _missing_old_text_error(store, target, "replace")
        return tool_error(f"{missing} is required for 'replace' action.", success=False)
    if action == "remove" and not old_text:
        return _missing_old_text_error(store, target, "remove")

    # Approval gate: when on, stages the write (background/gateway) or prompts
    # inline (interactive CLI); when off (default) passes straight through.
    gate_result = _apply_write_gate(action, target, content, old_text)
    if gate_result is not None:
        return gate_result

    if action == "add":
        result = store.add(target, content)

    elif action == "replace":
        result = store.replace(target, old_text, content)

    elif action == "remove":
        result = store.remove(target, old_text)

    else:
        return tool_error(f"Unknown action '{action}'. Use: add, replace, remove, search", success=False)

    return json.dumps(result, ensure_ascii=False)


def check_memory_requirements() -> bool:
    """Memory tool has no external requirements -- always available."""
    return True


def apply_memory_pending(payload: Dict[str, Any], store: "MemoryStore") -> Dict[str, Any]:
    """Replay a staged memory write directly against the store, bypassing the
    write gate. Called by the /memory approve handler.

    Returns the store's result dict.
    """
    action = payload.get("action")
    target = payload.get("target", "memory")
    content = payload.get("content") or ""
    old_text = payload.get("old_text") or ""
    if action == "batch":
        return store.apply_batch(target, payload.get("operations") or [])
    if action == "add":
        return store.add(target, content)
    if action == "replace":
        return store.replace(target, old_text, content)
    if action == "remove":
        return store.remove(target, old_text)
    return {"success": False, "error": f"Unknown staged action '{action}'."}
# OpenAI Function-Calling Schema
# =============================================================================

MEMORY_SCHEMA = {
    "name": "memory",
    "description": (
        "Save durable facts to persistent memory that survive across sessions. Memory has "
        "THREE TIERS per target: tier1 is injected into every future turn (keep it compact, "
        "high-signal); tier2/tier3 are retrieval-only overflow that never enters the prompt -- "
        "use action='search' to look them up. Adding to a full tier1 automatically cascades "
        "its oldest entries down into tier2, then tier3, so nothing is ever lost silently.\n\n"
        "HOW: make ALL your changes in ONE call via an 'operations' array (each item: "
        "{action, content?, old_text?}). The batch applies atomically and the char limit is "
        "checked only on the FINAL result — so a single call can remove/replace stale entries "
        "to free room AND add new ones, even when an add alone would overflow. The response "
        "reports current/limit chars and confirms completion; one batch call finishes the "
        "update, so don't repeat it. Use the bare action/content/old_text fields only for a "
        "single lone change. (operations does not support 'search' -- search is single-op only.)\n\n"
        "WHEN: save proactively when the user states a preference, correction, or personal "
        "detail, or you learn a stable fact about their environment, conventions, or workflow. "
        "Priority: user preferences & corrections > environment facts > procedures. The best "
        "memory stops the user repeating themselves.\n\n"
        "IF TIER1 IS FULL and cascading into tier2/tier3 also has no room (tier3 itself is full): "
        "the add is refused with 'eviction_candidates' from tier3. Before retrying, consider "
        "whether one of those candidates is really a reusable procedure that belongs in a skill "
        "(use skill_manage to fold it in), then memory(action='remove') it to free room -- or "
        "just remove it directly if it's simply stale. The tool never deletes tier3 content on "
        "its own.\n\n"
        "SEARCH: action='search' with a 'query' substring looks up tier2/tier3 (never tier1 -- "
        "it's already in your context, no need to search it). Optional 'tier' narrows to "
        "'tier2' or 'tier3' only (default 'all' = both). Read-only, does not mutate anything, "
        "results capped at 20 with a true total_matches count.\n\n"
        "TARGETS: 'user' = who the user is (name, role, preferences, style). 'memory' = your "
        "notes (environment, conventions, tool quirks, lessons).\n\n"
        "SKIP: trivial/obvious info, easily re-discovered facts, raw data dumps, task progress, "
        "completed-work logs, temporary TODO state (use session_search for those). Reusable "
        "procedures belong in a skill, not memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "replace", "remove", "search"],
                "description": "The action to perform (single-op shape). Omit when using 'operations' (which only supports add/replace/remove)."
            },
            "target": {
                "type": "string",
                "enum": ["memory", "user"],
                "description": "Which memory store: 'memory' for personal notes, 'user' for user profile."
            },
            "content": {
                "type": "string",
                "description": "The entry content. Required for 'add' and 'replace' (single-op shape)."
            },
            "old_text": {
                "type": "string",
                "description": "REQUIRED for 'replace' and 'remove' (single-op shape): a short unique substring identifying the existing entry to modify, searched across tier1/tier2/tier3. Omit only for 'add'."
            },
            "query": {
                "type": "string",
                "description": "REQUIRED for action='search': a case-insensitive substring to look up in tier2/tier3 (tier1 is never searched -- it's already in context)."
            },
            "tier": {
                "type": "string",
                "enum": ["tier2", "tier3", "all"],
                "description": "Optional, only used with action='search'. Narrows the search to one overflow tier. Default 'all' searches both tier2 and tier3."
            },
            "operations": {
                "type": "array",
                "description": (
                    "Batch shape: a list of operations applied atomically in one call "
                    "against the final char budget. Preferred when making multiple changes "
                    "or consolidating to make room. Each item is {action, content?, old_text?}. "
                    "Search is not supported in batch -- issue it as a separate single-op call."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["add", "replace", "remove"]},
                        "content": {"type": "string", "description": "Entry content for add/replace."},
                        "old_text": {"type": "string", "description": "Substring identifying the entry for replace/remove."},
                    },
                    "required": ["action"],
                },
            },
        },
        "required": ["target"],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="memory",
    toolset="memory",
    schema=MEMORY_SCHEMA,
    handler=lambda args, **kw: memory_tool(
        action=args.get("action", ""),
        target=args.get("target", "memory"),
        content=args.get("content"),
        old_text=args.get("old_text"),
        operations=args.get("operations"),
        query=args.get("query"),
        tier=args.get("tier", "all"),
        store=kw.get("store")),
    check_fn=check_memory_requirements,
    emoji="🧠",
)




