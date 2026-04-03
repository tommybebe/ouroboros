"""Worktree merge operations for parallel AC isolation.

Merges AC worktree branches back into the main task branch after parallel
execution completes. When git auto-merge succeeds (no overlapping hunks),
the merge completes automatically without agent intervention.

When git cannot auto-merge (conflicting hunks), the merge is aborted and
a MergeConflict result is returned so the caller can dispatch a merge-agent.

Architecture:
- Each isolated AC runs on branch ooo/{execution_id}_ac_{index}
- After execution, each AC's changes are committed on its branch
- Branches are merged sequentially into the task branch
- Sequential order ensures deterministic results and simplifies conflict detection
- The existing LevelCoordinator remains as a safety-net fallback layer

Usage:
    merger = WorktreeMerger(repo_root="/path/to/repo")

    # Merge a single AC branch (auto-merge path)
    result = merger.merge_ac_branch(
        target_branch="ooo/orch_abc123",
        ac_branch="ooo/orch_abc123_ac_0",
        ac_index=0,
    )

    if result.succeeded:
        print(f"Auto-merged AC {result.ac_index}")
    elif result.has_conflicts:
        print(f"Conflicts in: {result.conflicting_files}")
        # Caller dispatches merge-agent

    # Or merge all AC branches at once
    plan_result = merger.merge_all(
        target_branch="ooo/orch_abc123",
        ac_branches=[(0, "ooo/orch_abc123_ac_0"), (1, "ooo/orch_abc123_ac_1")],
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ouroboros.core.worktree import WorktreeError, _run_git, _run_git_process
from ouroboros.observability.logging import get_logger

log = get_logger(__name__)


class MergeOutcome(Enum):
    """Outcome of a single AC branch merge attempt."""

    SUCCESS = "success"
    """Git auto-merge succeeded — no agent intervention needed."""

    CONFLICT = "conflict"
    """Git could not auto-merge — conflicting hunks need resolution."""

    NOTHING_TO_MERGE = "nothing_to_merge"
    """AC branch has no changes relative to the target."""

    ERROR = "error"
    """Unexpected git error during merge."""


@dataclass(frozen=True, slots=True)
class MergeResult:
    """Result of merging a single AC branch into the target branch.

    Attributes:
        ac_index: 0-based AC index that was merged.
        ac_branch: Git branch name for the AC.
        outcome: Whether the merge succeeded, conflicted, or errored.
        merge_sha: Commit SHA after successful merge (None otherwise).
        conflicting_files: Files with merge conflicts (empty on success).
        conflict_diff: Raw conflict diff text for merge-agent (empty on success).
        error_message: Error details when outcome is ERROR.
        warnings: Non-fatal warnings emitted during merge.
    """

    ac_index: int
    ac_branch: str
    outcome: MergeOutcome
    merge_sha: str | None = None
    conflicting_files: tuple[str, ...] = field(default_factory=tuple)
    conflict_diff: str = ""
    error_message: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def succeeded(self) -> bool:
        """True when auto-merge completed without agent intervention."""
        return self.outcome == MergeOutcome.SUCCESS

    @property
    def has_conflicts(self) -> bool:
        """True when git could not auto-merge due to conflicting hunks."""
        return self.outcome == MergeOutcome.CONFLICT

    @property
    def is_noop(self) -> bool:
        """True when the AC had nothing to merge."""
        return self.outcome == MergeOutcome.NOTHING_TO_MERGE

    def to_dict(self) -> dict[str, Any]:
        """Serialize for event/checkpoint storage."""
        return {
            "ac_index": self.ac_index,
            "ac_branch": self.ac_branch,
            "outcome": self.outcome.value,
            "merge_sha": self.merge_sha,
            "conflicting_files": list(self.conflicting_files),
            "error_message": self.error_message,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class MergePlanResult:
    """Result of merging all AC branches for a parallel level.

    Attributes:
        results: Per-AC merge results in merge order.
        all_succeeded: True when every AC auto-merged without conflicts.
        conflict_results: Subset of results that need merge-agent intervention.
        warnings_for_next_level: Accumulated warnings for downstream verification.
    """

    results: tuple[MergeResult, ...] = field(default_factory=tuple)

    @property
    def all_succeeded(self) -> bool:
        """True when every AC branch auto-merged or had nothing to merge."""
        return all(r.succeeded or r.is_noop for r in self.results)

    @property
    def conflict_results(self) -> tuple[MergeResult, ...]:
        """Merge results that have unresolved conflicts."""
        return tuple(r for r in self.results if r.has_conflicts)

    @property
    def success_count(self) -> int:
        """Number of ACs that auto-merged successfully."""
        return sum(1 for r in self.results if r.succeeded)

    @property
    def conflict_count(self) -> int:
        """Number of ACs with merge conflicts."""
        return sum(1 for r in self.results if r.has_conflicts)

    @property
    def warnings_for_next_level(self) -> tuple[str, ...]:
        """Accumulated warnings from all merge results."""
        warnings: list[str] = []
        for r in self.results:
            warnings.extend(r.warnings)
        return tuple(warnings)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for event/checkpoint storage."""
        return {
            "results": [r.to_dict() for r in self.results],
            "all_succeeded": self.all_succeeded,
            "success_count": self.success_count,
            "conflict_count": self.conflict_count,
        }


class WorktreeMerger:
    """Merges AC worktree branches back into the task branch.

    Handles the auto-merge fast path: when git can merge without conflicts,
    the merge completes immediately without any agent intervention. Only
    when git reports conflicts does it signal the need for a merge-agent.

    Thread-safety: NOT thread-safe. Should be called from a single
    orchestrator coroutine after all parallel ACs have completed.
    """

    def __init__(self, repo_root: str | Path) -> None:
        """Initialize merger.

        Args:
            repo_root: Absolute path to the repository root.
        """
        self._repo_root = Path(repo_root).resolve()

    def _has_changes(self, target_branch: str, ac_branch: str) -> bool:
        """Check if ac_branch has commits not in target_branch."""
        result = _run_git_process(
            ["rev-list", "--count", f"{target_branch}..{ac_branch}"],
            self._repo_root,
        )
        if result.returncode != 0:
            return False
        count = result.stdout.strip()
        return count != "0"

    def _current_branch(self) -> str:
        """Get the currently checked-out branch name."""
        return _run_git(["rev-parse", "--abbrev-ref", "HEAD"], self._repo_root)

    def _checkout(self, branch: str) -> None:
        """Switch to the given branch."""
        _run_git(["checkout", branch], self._repo_root)

    def _get_conflicting_files(self) -> tuple[str, ...]:
        """List files with merge conflicts in the working tree."""
        result = _run_git_process(
            ["diff", "--name-only", "--diff-filter=U"],
            self._repo_root,
        )
        if result.returncode != 0:
            return ()
        files = [f.strip() for f in result.stdout.strip().splitlines() if f.strip()]
        return tuple(files)

    def _get_conflict_diff(self) -> str:
        """Get the raw diff showing conflict markers."""
        result = _run_git_process(["diff"], self._repo_root)
        if result.returncode != 0:
            return ""
        return result.stdout

    def merge_ac_branch(
        self,
        target_branch: str,
        ac_branch: str,
        ac_index: int,
    ) -> MergeResult:
        """Merge a single AC branch into the target branch.

        When git auto-merge succeeds (no overlapping hunks), the merge
        completes immediately — no agent intervention is needed.

        When git cannot auto-merge, the merge is aborted and a CONFLICT
        result is returned with the conflict details for merge-agent dispatch.

        Args:
            target_branch: Branch to merge into (the task branch).
            ac_branch: Branch to merge from (AC's worktree branch).
            ac_index: 0-based AC index for tracking.

        Returns:
            MergeResult describing the outcome.
        """
        try:
            return self._do_merge(target_branch, ac_branch, ac_index)
        except WorktreeError as exc:
            log.error(
                "worktree_merge.error",
                ac_index=ac_index,
                ac_branch=ac_branch,
                error=str(exc),
            )
            return MergeResult(
                ac_index=ac_index,
                ac_branch=ac_branch,
                outcome=MergeOutcome.ERROR,
                error_message=str(exc),
            )
        except Exception as exc:
            log.exception(
                "worktree_merge.unexpected_error",
                ac_index=ac_index,
                ac_branch=ac_branch,
            )
            return MergeResult(
                ac_index=ac_index,
                ac_branch=ac_branch,
                outcome=MergeOutcome.ERROR,
                error_message=f"Unexpected error: {exc}",
            )

    def _do_merge(
        self,
        target_branch: str,
        ac_branch: str,
        ac_index: int,
    ) -> MergeResult:
        """Internal merge implementation.

        Steps:
        1. Check if AC branch has any changes relative to target
        2. Ensure we're on the target branch
        3. Attempt git merge --no-edit (auto-merge)
        4. On success: return SUCCESS with merge SHA
        5. On conflict: capture conflict info, abort merge, return CONFLICT
        """
        # Step 1: Check if there's anything to merge
        if not self._has_changes(target_branch, ac_branch):
            log.info(
                "worktree_merge.nothing_to_merge",
                ac_index=ac_index,
                ac_branch=ac_branch,
            )
            return MergeResult(
                ac_index=ac_index,
                ac_branch=ac_branch,
                outcome=MergeOutcome.NOTHING_TO_MERGE,
            )

        # Step 2: Ensure we're on the target branch
        current = self._current_branch()
        if current != target_branch:
            self._checkout(target_branch)

        # Step 3: Attempt auto-merge
        merge_result = _run_git_process(
            ["merge", "--no-edit", "--no-ff", ac_branch,
             "-m", f"Merge AC {ac_index} ({ac_branch})"],
            self._repo_root,
        )

        # Step 4: Auto-merge succeeded
        if merge_result.returncode == 0:
            merge_sha = _run_git(["rev-parse", "HEAD"], self._repo_root)
            log.info(
                "worktree_merge.auto_merged",
                ac_index=ac_index,
                ac_branch=ac_branch,
                merge_sha=merge_sha[:12],
            )
            return MergeResult(
                ac_index=ac_index,
                ac_branch=ac_branch,
                outcome=MergeOutcome.SUCCESS,
                merge_sha=merge_sha,
            )

        # Step 5: Merge failed — check if it's a conflict or another error
        stderr = merge_result.stderr.strip()
        stdout = merge_result.stdout.strip()

        # Detect actual merge conflicts (vs other git errors)
        is_conflict = (
            "CONFLICT" in stdout
            or "CONFLICT" in stderr
            or "Automatic merge failed" in stdout
            or "Automatic merge failed" in stderr
        )

        if is_conflict:
            # Capture conflict details before aborting
            conflicting_files = self._get_conflicting_files()
            conflict_diff = self._get_conflict_diff()

            # Abort the failed merge to restore clean state
            _run_git_process(["merge", "--abort"], self._repo_root)

            log.warning(
                "worktree_merge.conflict_detected",
                ac_index=ac_index,
                ac_branch=ac_branch,
                conflicting_files=list(conflicting_files),
            )
            return MergeResult(
                ac_index=ac_index,
                ac_branch=ac_branch,
                outcome=MergeOutcome.CONFLICT,
                conflicting_files=conflicting_files,
                conflict_diff=conflict_diff,
            )

        # Non-conflict git error — abort if merge is in progress
        _run_git_process(["merge", "--abort"], self._repo_root)

        log.error(
            "worktree_merge.git_error",
            ac_index=ac_index,
            ac_branch=ac_branch,
            stderr=stderr,
            stdout=stdout,
        )
        return MergeResult(
            ac_index=ac_index,
            ac_branch=ac_branch,
            outcome=MergeOutcome.ERROR,
            error_message=f"Git merge failed: {stderr or stdout}",
        )

    def merge_all(
        self,
        target_branch: str,
        ac_branches: list[tuple[int, str]],
    ) -> MergePlanResult:
        """Merge all AC branches sequentially into the target branch.

        Merges are applied in order of ac_index for determinism. When an
        auto-merge succeeds, it proceeds to the next branch. When a conflict
        is encountered, it captures the conflict info and continues attempting
        remaining branches (each against the current target state).

        This allows the caller to batch all conflict results and dispatch
        merge-agents as needed, rather than stopping at the first conflict.

        Args:
            target_branch: Branch to merge into (the task branch).
            ac_branches: List of (ac_index, branch_name) pairs, merged in order.

        Returns:
            MergePlanResult with per-AC outcomes.
        """
        # Sort by ac_index for deterministic merge order
        sorted_branches = sorted(ac_branches, key=lambda x: x[0])
        results: list[MergeResult] = []

        log.info(
            "worktree_merge.plan_started",
            target_branch=target_branch,
            ac_count=len(sorted_branches),
        )

        for ac_index, ac_branch in sorted_branches:
            result = self.merge_ac_branch(target_branch, ac_branch, ac_index)
            results.append(result)

        plan = MergePlanResult(results=tuple(results))

        log.info(
            "worktree_merge.plan_completed",
            target_branch=target_branch,
            success_count=plan.success_count,
            conflict_count=plan.conflict_count,
            all_succeeded=plan.all_succeeded,
        )

        return plan


__all__ = [
    "MergeOutcome",
    "MergePlanResult",
    "MergeResult",
    "WorktreeMerger",
]
