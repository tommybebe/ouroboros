"""Tests for worktree-aware path resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from ouroboros.orchestrator.worktree_path_resolver import (
    WorktreePathResolver,
    build_resolver_for_ac,
    normalize_files_modified,
)


class TestSharedResolver:
    """SHARED-mode resolver returns paths unchanged (zero overhead)."""

    def test_shared_is_inactive(self) -> None:
        resolver = WorktreePathResolver.shared()
        assert resolver.is_active is False

    def test_to_worktree_noop(self) -> None:
        resolver = WorktreePathResolver.shared()
        assert resolver.to_worktree("/home/user/project/src/foo.py") == "/home/user/project/src/foo.py"

    def test_to_main_repo_noop(self) -> None:
        resolver = WorktreePathResolver.shared()
        assert resolver.to_main_repo("/any/path/file.py") == "/any/path/file.py"

    def test_normalize_tool_file_path_noop(self) -> None:
        resolver = WorktreePathResolver.shared()
        p = "/home/user/project/src/foo.py"
        assert resolver.normalize_tool_file_path("Edit", p) == p

    def test_translate_file_paths_noop(self) -> None:
        resolver = WorktreePathResolver.shared()
        paths = ("/a/b.py", "/c/d.py")
        assert resolver.translate_file_paths(paths) == paths

    def test_make_relative_passthrough(self) -> None:
        resolver = WorktreePathResolver.shared()
        assert resolver.make_relative("/some/random/path.py") == "/some/random/path.py"


class TestActiveResolver:
    """WORKTREE-mode resolver translates paths bidirectionally."""

    @pytest.fixture()
    def resolver(self, tmp_path: Path) -> WorktreePathResolver:
        repo = tmp_path / "project"
        repo.mkdir()
        wt = tmp_path / "worktrees" / "project" / "orch_abc_ac_1"
        wt.mkdir(parents=True)
        return WorktreePathResolver.for_worktree(
            repo_root=str(repo),
            worktree_root=str(wt),
        )

    def test_is_active(self, resolver: WorktreePathResolver) -> None:
        assert resolver.is_active is True

    def test_to_worktree_translates_repo_path(self, resolver: WorktreePathResolver) -> None:
        repo_root = resolver.repo_root
        wt_root = resolver.worktree_root
        result = resolver.to_worktree(f"{repo_root}/src/foo.py")
        assert result == f"{wt_root}/src/foo.py"

    def test_to_worktree_preserves_nested_paths(self, resolver: WorktreePathResolver) -> None:
        repo_root = resolver.repo_root
        wt_root = resolver.worktree_root
        result = resolver.to_worktree(f"{repo_root}/src/deeply/nested/module.py")
        assert result == f"{wt_root}/src/deeply/nested/module.py"

    def test_to_worktree_repo_root_itself(self, resolver: WorktreePathResolver) -> None:
        result = resolver.to_worktree(resolver.repo_root)
        assert result == resolver.worktree_root

    def test_to_worktree_outside_repo_unchanged(self, resolver: WorktreePathResolver) -> None:
        p = "/usr/lib/python3/site-packages/foo.py"
        assert resolver.to_worktree(p) == p

    def test_to_main_repo_translates_worktree_path(self, resolver: WorktreePathResolver) -> None:
        repo_root = resolver.repo_root
        wt_root = resolver.worktree_root
        result = resolver.to_main_repo(f"{wt_root}/src/bar.py")
        assert result == f"{repo_root}/src/bar.py"

    def test_to_main_repo_outside_worktree_unchanged(self, resolver: WorktreePathResolver) -> None:
        p = "/tmp/other/file.py"
        assert resolver.to_main_repo(p) == p

    def test_round_trip_repo_to_worktree_and_back(self, resolver: WorktreePathResolver) -> None:
        original = f"{resolver.repo_root}/src/models/user.py"
        wt_path = resolver.to_worktree(original)
        restored = resolver.to_main_repo(wt_path)
        assert restored == original

    def test_round_trip_worktree_to_repo_and_back(self, resolver: WorktreePathResolver) -> None:
        original = f"{resolver.worktree_root}/tests/test_auth.py"
        repo_path = resolver.to_main_repo(original)
        restored = resolver.to_worktree(repo_path)
        assert restored == original


class TestNormalizeToolFilePath:
    """Tool file path normalization for worktree execution."""

    @pytest.fixture()
    def resolver(self, tmp_path: Path) -> WorktreePathResolver:
        repo = tmp_path / "project"
        repo.mkdir()
        wt = tmp_path / "worktrees" / "project" / "orch_abc_ac_1"
        wt.mkdir(parents=True)
        return WorktreePathResolver.for_worktree(
            repo_root=str(repo),
            worktree_root=str(wt),
        )

    def test_main_repo_path_translated_to_worktree(self, resolver: WorktreePathResolver) -> None:
        repo_path = f"{resolver.repo_root}/src/foo.py"
        result = resolver.normalize_tool_file_path("Edit", repo_path)
        assert result == f"{resolver.worktree_root}/src/foo.py"

    def test_worktree_path_unchanged(self, resolver: WorktreePathResolver) -> None:
        wt_path = f"{resolver.worktree_root}/src/foo.py"
        result = resolver.normalize_tool_file_path("Read", wt_path)
        assert result == wt_path

    def test_external_path_unchanged(self, resolver: WorktreePathResolver) -> None:
        p = "/usr/local/lib/something.py"
        result = resolver.normalize_tool_file_path("Write", p)
        assert result == p

    def test_works_for_all_file_tools(self, resolver: WorktreePathResolver) -> None:
        repo_path = f"{resolver.repo_root}/config.yaml"
        for tool in ("Read", "Edit", "Write", "NotebookEdit"):
            result = resolver.normalize_tool_file_path(tool, repo_path)
            assert result == f"{resolver.worktree_root}/config.yaml"


class TestBatchTranslation:
    """Test translate_file_paths batch operation."""

    @pytest.fixture()
    def resolver(self, tmp_path: Path) -> WorktreePathResolver:
        repo = tmp_path / "project"
        repo.mkdir()
        wt = tmp_path / "worktrees" / "project" / "orch_abc_ac_1"
        wt.mkdir(parents=True)
        return WorktreePathResolver.for_worktree(
            repo_root=str(repo),
            worktree_root=str(wt),
        )

    def test_batch_to_worktree(self, resolver: WorktreePathResolver) -> None:
        paths = (
            f"{resolver.repo_root}/a.py",
            f"{resolver.repo_root}/b.py",
        )
        result = resolver.translate_file_paths(paths, to_worktree=True)
        assert result == (
            f"{resolver.worktree_root}/a.py",
            f"{resolver.worktree_root}/b.py",
        )

    def test_batch_to_main(self, resolver: WorktreePathResolver) -> None:
        paths = (
            f"{resolver.worktree_root}/a.py",
            f"{resolver.worktree_root}/b.py",
        )
        result = resolver.translate_file_paths(paths, to_worktree=False)
        assert result == (
            f"{resolver.repo_root}/a.py",
            f"{resolver.repo_root}/b.py",
        )

    def test_preserves_order(self, resolver: WorktreePathResolver) -> None:
        paths = [
            f"{resolver.repo_root}/z.py",
            f"{resolver.repo_root}/a.py",
            f"{resolver.repo_root}/m.py",
        ]
        result = resolver.translate_file_paths(paths)
        assert [Path(p).name for p in result] == ["z.py", "a.py", "m.py"]

    def test_empty_returns_empty(self, resolver: WorktreePathResolver) -> None:
        assert resolver.translate_file_paths(()) == ()


class TestMakeRelative:
    """Test make_relative for repo-relative path extraction."""

    @pytest.fixture()
    def resolver(self, tmp_path: Path) -> WorktreePathResolver:
        repo = tmp_path / "project"
        repo.mkdir()
        wt = tmp_path / "worktrees" / "project" / "orch_abc_ac_1"
        wt.mkdir(parents=True)
        return WorktreePathResolver.for_worktree(
            repo_root=str(repo),
            worktree_root=str(wt),
        )

    def test_worktree_path_to_relative(self, resolver: WorktreePathResolver) -> None:
        result = resolver.make_relative(f"{resolver.worktree_root}/src/foo.py")
        assert result == "src/foo.py"

    def test_main_repo_path_to_relative(self, resolver: WorktreePathResolver) -> None:
        result = resolver.make_relative(f"{resolver.repo_root}/src/foo.py")
        assert result == "src/foo.py"

    def test_external_path_unchanged(self, resolver: WorktreePathResolver) -> None:
        p = "/usr/lib/something.py"
        assert resolver.make_relative(p) == p

    def test_root_paths_become_dot(self, resolver: WorktreePathResolver) -> None:
        result = resolver.make_relative(resolver.worktree_root)
        assert result == "."


# --- Factory function tests ---


@dataclass
class _FakeWorktreeInfo:
    """Minimal duck-type stand-in for ACWorktreeInfo."""

    worktree_path: str


class TestBuildResolverForAC:
    """Test build_resolver_for_ac factory function."""

    def test_shared_mode_returns_inactive(self, tmp_path: Path) -> None:
        resolver = build_resolver_for_ac(
            repo_root=str(tmp_path),
            worktree_info=None,
            isolation_mode="shared",
        )
        assert resolver.is_active is False

    def test_worktree_mode_returns_active(self, tmp_path: Path) -> None:
        repo = tmp_path / "project"
        repo.mkdir()
        wt = tmp_path / "wt"
        wt.mkdir()
        info = _FakeWorktreeInfo(worktree_path=str(wt))
        resolver = build_resolver_for_ac(
            repo_root=str(repo),
            worktree_info=info,
            isolation_mode="worktree",
        )
        assert resolver.is_active is True

    def test_worktree_mode_without_info_falls_back_to_shared(self, tmp_path: Path) -> None:
        resolver = build_resolver_for_ac(
            repo_root=str(tmp_path),
            worktree_info=None,
            isolation_mode="worktree",
        )
        assert resolver.is_active is False

    def test_worktree_mode_with_empty_path_falls_back(self, tmp_path: Path) -> None:
        info = _FakeWorktreeInfo(worktree_path="")
        resolver = build_resolver_for_ac(
            repo_root=str(tmp_path),
            worktree_info=info,
            isolation_mode="worktree",
        )
        assert resolver.is_active is False


class TestNormalizeFilesModified:
    """Test normalize_files_modified helper."""

    def test_shared_resolver_returns_unchanged(self) -> None:
        resolver = WorktreePathResolver.shared()
        paths = ("/repo/a.py", "/repo/b.py")
        assert normalize_files_modified(paths, resolver) == paths

    def test_active_resolver_translates_worktree_to_main(self, tmp_path: Path) -> None:
        repo = tmp_path / "project"
        repo.mkdir()
        wt = tmp_path / "wt"
        wt.mkdir()
        resolver = WorktreePathResolver.for_worktree(
            repo_root=str(repo),
            worktree_root=str(wt),
        )
        wt_paths = (f"{resolver.worktree_root}/src/a.py", f"{resolver.worktree_root}/src/b.py")
        result = normalize_files_modified(wt_paths, resolver)
        assert result == (f"{resolver.repo_root}/src/a.py", f"{resolver.repo_root}/src/b.py")

    def test_empty_tuple_returns_empty(self, tmp_path: Path) -> None:
        repo = tmp_path / "project"
        repo.mkdir()
        wt = tmp_path / "wt"
        wt.mkdir()
        resolver = WorktreePathResolver.for_worktree(
            repo_root=str(repo),
            worktree_root=str(wt),
        )
        assert normalize_files_modified((), resolver) == ()
