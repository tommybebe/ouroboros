"""Per-AC git worktree lifecycle management for parallel execution isolation.

Extends the core worktree system (core/worktree.py) to per-AC granularity,
providing create/track/cleanup operations for ACs that need file-level
isolation during parallel execution.

ACs with no predicted file overlap continue in the shared workspace with zero
overhead. Only ACs flagged for isolation get their own worktree.

Worktree branches follow the naming convention: ooo/{execution_id}_ac_{index}

Usage:
    manager = ACWorktreeManager(execution_id="orch_abc123", repo_root="/path/to/repo")

    # Create isolated worktree for an AC
    workspace = manager.create_ac_worktree(ac_index=2)

    # Track active worktrees
    info = manager.get_worktree(ac_index=2)

    # Cleanup after execution
    manager.remove_ac_worktree(ac_index=2)

    # Or cleanup all at once
    manager.remove_all()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ouroboros.core.worktree import (
    WorktreeError,
    _branch_exists,
    _ensure_worktree,
    _run_git,
    _run_git_process,
    _worktree_root,
)
from ouroboros.observability.logging import get_logger

log = get_logger(__name__)


class IsolationMode(Enum):
    """Execution isolation mode for an AC."""

    SHARED = "shared"
    """AC runs in the shared workspace (no overhead)."""

    WORKTREE = "worktree"
    """AC runs in its own git worktree for file isolation."""


@dataclass(frozen=True, slots=True)
class ACWorktreeInfo:
    """Metadata for an AC's isolated worktree.

    Attributes:
        ac_index: 0-based AC index.
        execution_id: Parent execution identifier.
        branch: Git branch name (ooo/{execution_id}_ac_{index}).
        worktree_path: Absolute path to the worktree directory.
        effective_cwd: Working directory inside the worktree (preserves subdir offset).
        isolation_mode: Always WORKTREE for tracked entries.
    """

    ac_index: int
    execution_id: str
    branch: str
    worktree_path: str
    effective_cwd: str
    isolation_mode: IsolationMode = IsolationMode.WORKTREE

    def to_dict(self) -> dict[str, Any]:
        """Serialize for checkpoint persistence."""
        return {
            "ac_index": self.ac_index,
            "execution_id": self.execution_id,
            "branch": self.branch,
            "worktree_path": self.worktree_path,
            "effective_cwd": self.effective_cwd,
            "isolation_mode": self.isolation_mode.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ACWorktreeInfo | None:
        """Deserialize from checkpoint data. Returns None on invalid input."""
        required = {"ac_index", "execution_id", "branch", "worktree_path", "effective_cwd"}
        if not required.issubset(data):
            return None
        try:
            mode = IsolationMode(data.get("isolation_mode", "worktree"))
        except ValueError:
            mode = IsolationMode.WORKTREE
        return cls(
            ac_index=data["ac_index"],
            execution_id=data["execution_id"],
            branch=data["branch"],
            worktree_path=data["worktree_path"],
            effective_cwd=data["effective_cwd"],
            isolation_mode=mode,
        )


@dataclass(slots=True)
class ACWorktreeManager:
    """Manages per-AC worktree lifecycle for a single parallel execution.

    Thread-safety: This class is NOT thread-safe. It should be used from a
    single orchestrator coroutine that dispatches AC execution. The git
    operations themselves are process-safe via filesystem locking.

    Attributes:
        execution_id: Unique execution identifier (e.g. "orch_abc123").
        repo_root: Absolute path to the main repository root.
        source_cwd: Original working directory (for subdir offset calculation).
        _active: Mutable tracking dict of ac_index -> ACWorktreeInfo.
    """

    execution_id: str
    repo_root: str
    source_cwd: str
    _active: dict[int, ACWorktreeInfo] = field(default_factory=dict, repr=False)

    def _branch_name(self, ac_index: int) -> str:
        """Build the worktree branch name for an AC."""
        return f"ooo/{self.execution_id}_ac_{ac_index}"

    def _worktree_path(self, ac_index: int) -> Path:
        """Build the worktree filesystem path for an AC."""
        repo_name = Path(self.repo_root).name
        wt_id = f"{self.execution_id}_ac_{ac_index}"
        return _worktree_root() / repo_name / wt_id

    def _effective_cwd(self, worktree_path: Path) -> str:
        """Calculate effective cwd inside worktree (preserving subdir offset)."""
        repo_root = Path(self.repo_root).resolve()
        source = Path(self.source_cwd).resolve()
        try:
            relative = source.relative_to(repo_root)
        except ValueError:
            # source_cwd is not inside repo_root; use worktree root
            return str(worktree_path)
        return str(worktree_path / relative)

    def create_ac_worktree(
        self,
        ac_index: int,
        *,
        base_ref: str | None = None,
    ) -> ACWorktreeInfo:
        """Create an isolated git worktree for an AC.

        Creates a new worktree branching from the current HEAD (or base_ref)
        and registers it in the tracking structure.

        Args:
            ac_index: 0-based AC index.
            base_ref: Optional git ref to branch from. Defaults to HEAD.

        Returns:
            ACWorktreeInfo with paths and branch metadata.

        Raises:
            WorktreeError: If the worktree already exists or git operations fail.
        """
        if ac_index in self._active:
            raise WorktreeError(
                "AC worktree already exists",
                details={
                    "ac_index": ac_index,
                    "execution_id": self.execution_id,
                    "existing_path": self._active[ac_index].worktree_path,
                },
            )

        branch = self._branch_name(ac_index)
        wt_path = self._worktree_path(ac_index)
        repo_root = Path(self.repo_root)

        # Validate branch name
        result = _run_git_process(["check-ref-format", "--branch", branch], repo_root)
        if result.returncode != 0:
            raise WorktreeError(
                "Invalid branch name for AC worktree",
                details={
                    "branch": branch,
                    "execution_id": self.execution_id,
                    "ac_index": ac_index,
                },
            )

        # Create the worktree (reuses existing branch if present)
        _ensure_worktree(repo_root, wt_path, branch, base_ref=base_ref)

        effective_cwd = self._effective_cwd(wt_path)

        info = ACWorktreeInfo(
            ac_index=ac_index,
            execution_id=self.execution_id,
            branch=branch,
            worktree_path=str(wt_path),
            effective_cwd=effective_cwd,
        )
        self._active[ac_index] = info

        log.info(
            "ac_worktree.created",
            ac_index=ac_index,
            execution_id=self.execution_id,
            branch=branch,
            worktree_path=str(wt_path),
        )
        return info

    def get_worktree(self, ac_index: int) -> ACWorktreeInfo | None:
        """Look up active worktree info for an AC.

        Args:
            ac_index: 0-based AC index.

        Returns:
            ACWorktreeInfo if the AC has an active worktree, None otherwise.
        """
        return self._active.get(ac_index)

    @property
    def active_worktrees(self) -> dict[int, ACWorktreeInfo]:
        """Return a snapshot of all active AC worktrees."""
        return dict(self._active)

    @property
    def active_count(self) -> int:
        """Number of currently active AC worktrees."""
        return len(self._active)

    def remove_ac_worktree(self, ac_index: int, *, force: bool = False) -> bool:
        """Remove an AC's worktree and clean up its branch.

        Removes the git worktree and deletes the branch. The branch is only
        deleted if it has been merged or force=True.

        Args:
            ac_index: 0-based AC index.
            force: Force removal even if branch has unmerged changes.

        Returns:
            True if the worktree was removed, False if it wasn't tracked.
        """
        info = self._active.pop(ac_index, None)
        if info is None:
            return False

        repo_root = Path(self.repo_root)
        wt_path = Path(info.worktree_path)

        # Remove the worktree
        try:
            remove_args = ["worktree", "remove"]
            if force:
                remove_args.append("--force")
            remove_args.append(str(wt_path))
            _run_git(remove_args, repo_root)
        except WorktreeError as exc:
            log.warning(
                "ac_worktree.remove_failed",
                ac_index=ac_index,
                execution_id=self.execution_id,
                error=str(exc),
            )
            # Fall back to force removal if normal removal failed
            if not force:
                try:
                    _run_git(["worktree", "remove", "--force", str(wt_path)], repo_root)
                except WorktreeError:
                    log.warning(
                        "ac_worktree.force_remove_failed",
                        ac_index=ac_index,
                        worktree_path=str(wt_path),
                    )

        # Prune stale worktree references
        try:
            _run_git(["worktree", "prune"], repo_root)
        except WorktreeError:
            pass

        # Delete the branch (only if not checked out elsewhere)
        branch = info.branch
        if _branch_exists(repo_root, branch):
            try:
                delete_flag = "-D" if force else "-d"
                _run_git(["branch", delete_flag, branch], repo_root)
            except WorktreeError as exc:
                log.warning(
                    "ac_worktree.branch_delete_failed",
                    branch=branch,
                    error=str(exc),
                )

        log.info(
            "ac_worktree.removed",
            ac_index=ac_index,
            execution_id=self.execution_id,
            branch=branch,
        )
        return True

    def remove_all(self, *, force: bool = False) -> int:
        """Remove all active AC worktrees.

        Args:
            force: Force removal even if branches have unmerged changes.

        Returns:
            Number of worktrees successfully removed.
        """
        indices = list(self._active.keys())
        removed = 0
        for ac_index in indices:
            if self.remove_ac_worktree(ac_index, force=force):
                removed += 1
        return removed

    def to_checkpoint(self) -> list[dict[str, Any]]:
        """Serialize active worktree state for checkpoint persistence."""
        return [info.to_dict() for info in self._active.values()]

    @classmethod
    def from_checkpoint(
        cls,
        execution_id: str,
        repo_root: str,
        source_cwd: str,
        data: list[dict[str, Any]],
    ) -> ACWorktreeManager:
        """Restore manager state from checkpoint data.

        Args:
            execution_id: Execution identifier.
            repo_root: Repository root path.
            source_cwd: Original working directory.
            data: Serialized worktree info list.

        Returns:
            Restored ACWorktreeManager with active worktrees re-populated.
        """
        manager = cls(
            execution_id=execution_id,
            repo_root=repo_root,
            source_cwd=source_cwd,
        )
        for entry in data:
            info = ACWorktreeInfo.from_dict(entry)
            if info is not None:
                # Only restore if the worktree directory still exists
                if Path(info.worktree_path).exists():
                    manager._active[info.ac_index] = info
                else:
                    log.warning(
                        "ac_worktree.restore_skipped",
                        ac_index=info.ac_index,
                        worktree_path=info.worktree_path,
                        reason="directory_missing",
                    )
        return manager

    def commit_ac_changes(self, ac_index: int, message: str) -> str | None:
        """Create a commit in an AC's worktree with all staged and unstaged changes.

        Args:
            ac_index: 0-based AC index.
            message: Commit message.

        Returns:
            The commit SHA, or None if there was nothing to commit.

        Raises:
            WorktreeError: If the AC has no active worktree.
        """
        info = self._active.get(ac_index)
        if info is None:
            raise WorktreeError(
                "No active worktree for AC",
                details={"ac_index": ac_index, "execution_id": self.execution_id},
            )

        wt_path = Path(info.worktree_path)

        # Check if there are any changes
        status = _run_git(["status", "--porcelain"], wt_path)
        if not status:
            return None

        # Stage all changes and commit
        _run_git(["add", "-A"], wt_path)
        _run_git(["commit", "-m", message], wt_path)
        sha = _run_git(["rev-parse", "HEAD"], wt_path)

        log.info(
            "ac_worktree.committed",
            ac_index=ac_index,
            execution_id=self.execution_id,
            sha=sha[:12],
        )
        return sha


__all__ = [
    "ACWorktreeInfo",
    "ACWorktreeManager",
    "IsolationMode",
]
