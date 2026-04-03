"""Worktree-aware path resolution for isolated AC execution.

Provides bidirectional path translation between main repo and worktree
directories so that agents executing in a worktree transparently read/write
files relative to their worktree root rather than the main repository.

Key invariant: SHARED-mode ACs bypass the resolver entirely (returns paths
unchanged) — zero overhead for the common case.

Usage:
    resolver = WorktreePathResolver.for_worktree(
        repo_root="/home/user/project",
        worktree_root="/home/user/.ouroboros/worktrees/project/orch_abc_ac_1",
    )

    # Main repo path → worktree path
    wt = resolver.to_worktree("/home/user/project/src/foo.py")
    # → "/home/user/.ouroboros/worktrees/project/orch_abc_ac_1/src/foo.py"

    # Worktree path → main repo path
    main = resolver.to_main_repo(wt)
    # → "/home/user/project/src/foo.py"

    # SHARED mode (no-op resolver)
    noop = WorktreePathResolver.shared()
    noop.to_worktree("/any/path")  # → "/any/path" unchanged
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ouroboros.observability.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class WorktreePathResolver:
    """Bidirectional path translator between main repo and worktree.

    When ``is_active`` is False (SHARED mode), all methods return paths
    unchanged — this is the zero-overhead fast path for ACs with no
    predicted file overlap.

    Attributes:
        repo_root: Resolved absolute path to the main repository root.
        worktree_root: Resolved absolute path to the AC's worktree root.
        is_active: True when worktree isolation is in effect.
    """

    repo_root: str
    worktree_root: str
    is_active: bool

    @classmethod
    def shared(cls) -> WorktreePathResolver:
        """Create a no-op resolver for SHARED-mode ACs.

        All path translation methods return the input path unchanged.
        """
        return cls(repo_root="", worktree_root="", is_active=False)

    @classmethod
    def for_worktree(
        cls,
        repo_root: str,
        worktree_root: str,
    ) -> WorktreePathResolver:
        """Create an active resolver for a WORKTREE-mode AC.

        Args:
            repo_root: Absolute path to the main repository root.
            worktree_root: Absolute path to this AC's worktree root.

        Returns:
            Active resolver that translates paths between main repo and worktree.
        """
        resolved_repo = str(Path(repo_root).resolve())
        resolved_wt = str(Path(worktree_root).resolve())
        return cls(
            repo_root=resolved_repo,
            worktree_root=resolved_wt,
            is_active=True,
        )

    def to_worktree(self, path: str) -> str:
        """Translate a main-repo absolute path to its worktree equivalent.

        If the path is not inside the main repo root, or the resolver is
        inactive (SHARED mode), the path is returned unchanged.

        Args:
            path: Absolute file path (typically from the main repo).

        Returns:
            Equivalent path inside the worktree, or the original path if
            translation is not applicable.
        """
        if not self.is_active:
            return path

        resolved = str(Path(path).resolve())
        try:
            relative = str(Path(resolved).relative_to(self.repo_root))
        except ValueError:
            # Path is not inside the main repo — return unchanged
            return path

        return str(Path(self.worktree_root) / relative)

    def to_main_repo(self, path: str) -> str:
        """Translate a worktree absolute path back to its main-repo equivalent.

        Used for reporting, merge operations, and context passing where
        downstream consumers expect main-repo-relative paths.

        If the path is not inside the worktree root, or the resolver is
        inactive (SHARED mode), the path is returned unchanged.

        Args:
            path: Absolute file path (typically from the worktree).

        Returns:
            Equivalent path inside the main repo, or the original path if
            translation is not applicable.
        """
        if not self.is_active:
            return path

        resolved = str(Path(path).resolve())
        try:
            relative = str(Path(resolved).relative_to(self.worktree_root))
        except ValueError:
            # Path is not inside the worktree — return unchanged
            return path

        return str(Path(self.repo_root) / relative)

    def translate_file_paths(
        self,
        paths: tuple[str, ...] | list[str],
        *,
        to_worktree: bool = True,
    ) -> tuple[str, ...]:
        """Batch-translate a collection of file paths.

        Args:
            paths: Sequence of absolute file paths.
            to_worktree: If True, translate main→worktree. If False, worktree→main.

        Returns:
            Tuple of translated paths (preserves ordering).
        """
        if not self.is_active:
            return tuple(paths)

        fn = self.to_worktree if to_worktree else self.to_main_repo
        return tuple(fn(p) for p in paths)

    def normalize_tool_file_path(self, tool_name: str, file_path: str) -> str:
        """Ensure a tool's file_path argument targets the correct workspace.

        When an AC runs in a worktree, tool invocations may still reference
        main-repo paths (e.g., from Glob results or hardcoded paths in prompts).
        This method normalizes such paths to point at the worktree copy.

        For SHARED-mode ACs, returns the path unchanged.

        Args:
            tool_name: Name of the tool (Read, Edit, Write, etc.).
            file_path: The file_path argument from the tool invocation.

        Returns:
            Normalized file path targeting the correct workspace.
        """
        if not self.is_active:
            return file_path

        # Only translate if the path is inside the main repo
        # (paths already inside the worktree are left alone)
        resolved = str(Path(file_path).resolve())

        # Already targeting worktree — no translation needed
        try:
            Path(resolved).relative_to(self.worktree_root)
            return file_path
        except ValueError:
            pass

        # Targeting main repo — translate to worktree
        try:
            Path(resolved).relative_to(self.repo_root)
            translated = self.to_worktree(file_path)
            log.debug(
                "worktree_path_resolver.translated",
                tool=tool_name,
                original=file_path,
                translated=translated,
            )
            return translated
        except ValueError:
            # Path is outside both — return unchanged
            return file_path

    def make_relative(self, path: str) -> str:
        """Convert an absolute path to a repo-relative path.

        Works with paths from either the main repo or the worktree.
        Returns the path unchanged if it's not inside either root.
        """
        resolved = str(Path(path).resolve())

        # Try worktree first (more specific)
        if self.is_active:
            try:
                return str(Path(resolved).relative_to(self.worktree_root))
            except ValueError:
                pass

        # Try main repo
        if self.repo_root:
            try:
                return str(Path(resolved).relative_to(self.repo_root))
            except ValueError:
                pass

        return path


def build_resolver_for_ac(
    repo_root: str,
    worktree_info: object | None,
    isolation_mode: str = "shared",
) -> WorktreePathResolver:
    """Build a path resolver based on AC isolation mode and worktree info.

    Convenience factory that integrates with ACWorktreeInfo and IsolationMode
    from the orchestrator layer. Returns a SHARED (no-op) resolver when the
    AC runs in the shared workspace, and an active resolver when it runs in
    a worktree.

    Args:
        repo_root: Absolute path to the main repository root.
        worktree_info: ACWorktreeInfo instance (or None for shared ACs).
            Duck-typed: expects ``worktree_path`` attribute.
        isolation_mode: "shared" or "worktree".

    Returns:
        Appropriate WorktreePathResolver instance.
    """
    if isolation_mode != "worktree" or worktree_info is None:
        return WorktreePathResolver.shared()

    wt_path = getattr(worktree_info, "worktree_path", None)
    if not wt_path:
        return WorktreePathResolver.shared()

    return WorktreePathResolver.for_worktree(
        repo_root=repo_root,
        worktree_root=str(wt_path),
    )


def normalize_files_modified(
    files_modified: tuple[str, ...],
    resolver: WorktreePathResolver,
) -> tuple[str, ...]:
    """Normalize worktree file paths back to main-repo-relative form.

    When an AC runs in a worktree, its tool calls record absolute paths
    inside the worktree directory. For context passing and conflict detection,
    these paths should be translated back to main-repo equivalents.

    For SHARED-mode ACs, paths are returned unchanged.

    Args:
        files_modified: Tuple of absolute file paths from tool call records.
        resolver: Path resolver for the AC that produced these paths.

    Returns:
        Tuple of normalized paths (main-repo absolute paths).
    """
    if not resolver.is_active:
        return files_modified
    return resolver.translate_file_paths(files_modified, to_worktree=False)


__all__ = [
    "WorktreePathResolver",
    "build_resolver_for_ac",
    "normalize_files_modified",
]
