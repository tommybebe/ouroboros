"""Tests for per-AC git worktree lifecycle management."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ouroboros.core.worktree import WorktreeError
from ouroboros.orchestrator.ac_worktree import (
    ACWorktreeInfo,
    ACWorktreeManager,
    IsolationMode,
)


@pytest.fixture()
def repo_root(tmp_path: Path) -> Path:
    """Create a minimal git repo for testing."""
    return tmp_path / "repo"


@pytest.fixture()
def manager(repo_root: Path) -> ACWorktreeManager:
    """Create an ACWorktreeManager with test defaults."""
    return ACWorktreeManager(
        execution_id="orch_abc123",
        repo_root=str(repo_root),
        source_cwd=str(repo_root),
    )


class TestACWorktreeInfoSerialization:
    """Test ACWorktreeInfo to_dict/from_dict round-trip."""

    def test_round_trip(self) -> None:
        info = ACWorktreeInfo(
            ac_index=2,
            execution_id="orch_abc123",
            branch="ooo/orch_abc123_ac_2",
            worktree_path="/tmp/wt/repo/orch_abc123_ac_2",
            effective_cwd="/tmp/wt/repo/orch_abc123_ac_2/src",
            isolation_mode=IsolationMode.WORKTREE,
        )
        data = info.to_dict()
        restored = ACWorktreeInfo.from_dict(data)

        assert restored is not None
        assert restored.ac_index == info.ac_index
        assert restored.execution_id == info.execution_id
        assert restored.branch == info.branch
        assert restored.worktree_path == info.worktree_path
        assert restored.effective_cwd == info.effective_cwd
        assert restored.isolation_mode == IsolationMode.WORKTREE

    def test_from_dict_missing_fields_returns_none(self) -> None:
        assert ACWorktreeInfo.from_dict({"ac_index": 0}) is None

    def test_from_dict_invalid_isolation_mode_defaults(self) -> None:
        data = {
            "ac_index": 0,
            "execution_id": "test",
            "branch": "ooo/test_ac_0",
            "worktree_path": "/tmp/wt",
            "effective_cwd": "/tmp/wt",
            "isolation_mode": "invalid_mode",
        }
        info = ACWorktreeInfo.from_dict(data)
        assert info is not None
        assert info.isolation_mode == IsolationMode.WORKTREE


class TestACWorktreeManagerBranchNaming:
    """Test branch name and path generation."""

    def test_branch_name_follows_convention(self, manager: ACWorktreeManager) -> None:
        assert manager._branch_name(0) == "ooo/orch_abc123_ac_0"
        assert manager._branch_name(3) == "ooo/orch_abc123_ac_3"

    def test_worktree_path_includes_repo_name(self, manager: ACWorktreeManager) -> None:
        with patch("ouroboros.orchestrator.ac_worktree._worktree_root") as mock_root:
            mock_root.return_value = Path("/home/user/.ouroboros/worktrees")
            path = manager._worktree_path(2)
        assert path == Path("/home/user/.ouroboros/worktrees/repo/orch_abc123_ac_2")


class TestACWorktreeManagerCreate:
    """Test worktree creation."""

    @patch("ouroboros.orchestrator.ac_worktree._ensure_worktree")
    @patch("ouroboros.orchestrator.ac_worktree._run_git_process")
    @patch("ouroboros.orchestrator.ac_worktree._worktree_root")
    def test_create_registers_and_returns_info(
        self,
        mock_root: MagicMock,
        mock_git_process: MagicMock,
        mock_ensure: MagicMock,
        manager: ACWorktreeManager,
        tmp_path: Path,
    ) -> None:
        mock_root.return_value = tmp_path / "worktrees"
        mock_git_process.return_value = MagicMock(returncode=0)

        info = manager.create_ac_worktree(ac_index=1)

        assert info.ac_index == 1
        assert info.branch == "ooo/orch_abc123_ac_1"
        assert info.isolation_mode == IsolationMode.WORKTREE
        assert manager.active_count == 1
        assert manager.get_worktree(1) is info
        mock_ensure.assert_called_once()

    @patch("ouroboros.orchestrator.ac_worktree._ensure_worktree")
    @patch("ouroboros.orchestrator.ac_worktree._run_git_process")
    @patch("ouroboros.orchestrator.ac_worktree._worktree_root")
    def test_create_duplicate_raises(
        self,
        mock_root: MagicMock,
        mock_git_process: MagicMock,
        mock_ensure: MagicMock,
        manager: ACWorktreeManager,
        tmp_path: Path,
    ) -> None:
        mock_root.return_value = tmp_path / "worktrees"
        mock_git_process.return_value = MagicMock(returncode=0)

        manager.create_ac_worktree(ac_index=0)

        with pytest.raises(WorktreeError, match="AC worktree already exists"):
            manager.create_ac_worktree(ac_index=0)

    @patch("ouroboros.orchestrator.ac_worktree._run_git_process")
    @patch("ouroboros.orchestrator.ac_worktree._worktree_root")
    def test_create_invalid_branch_raises(
        self,
        mock_root: MagicMock,
        mock_git_process: MagicMock,
        manager: ACWorktreeManager,
        tmp_path: Path,
    ) -> None:
        mock_root.return_value = tmp_path / "worktrees"
        mock_git_process.return_value = MagicMock(returncode=1)

        with pytest.raises(WorktreeError, match="Invalid branch name"):
            manager.create_ac_worktree(ac_index=0)

    @patch("ouroboros.orchestrator.ac_worktree._ensure_worktree")
    @patch("ouroboros.orchestrator.ac_worktree._run_git_process")
    @patch("ouroboros.orchestrator.ac_worktree._worktree_root")
    def test_create_with_base_ref(
        self,
        mock_root: MagicMock,
        mock_git_process: MagicMock,
        mock_ensure: MagicMock,
        manager: ACWorktreeManager,
        tmp_path: Path,
    ) -> None:
        mock_root.return_value = tmp_path / "worktrees"
        mock_git_process.return_value = MagicMock(returncode=0)

        manager.create_ac_worktree(ac_index=0, base_ref="abc123")

        mock_ensure.assert_called_once()
        _, kwargs = mock_ensure.call_args
        assert kwargs.get("base_ref") == "abc123"


class TestACWorktreeManagerRemove:
    """Test worktree removal."""

    @patch("ouroboros.orchestrator.ac_worktree._branch_exists", return_value=True)
    @patch("ouroboros.orchestrator.ac_worktree._run_git")
    @patch("ouroboros.orchestrator.ac_worktree._ensure_worktree")
    @patch("ouroboros.orchestrator.ac_worktree._run_git_process")
    @patch("ouroboros.orchestrator.ac_worktree._worktree_root")
    def test_remove_cleans_up(
        self,
        mock_root: MagicMock,
        mock_git_process: MagicMock,
        mock_ensure: MagicMock,
        mock_run_git: MagicMock,
        mock_branch_exists: MagicMock,
        manager: ACWorktreeManager,
        tmp_path: Path,
    ) -> None:
        mock_root.return_value = tmp_path / "worktrees"
        mock_git_process.return_value = MagicMock(returncode=0)

        manager.create_ac_worktree(ac_index=1)
        assert manager.active_count == 1

        result = manager.remove_ac_worktree(ac_index=1)

        assert result is True
        assert manager.active_count == 0
        assert manager.get_worktree(1) is None

    def test_remove_nonexistent_returns_false(self, manager: ACWorktreeManager) -> None:
        assert manager.remove_ac_worktree(ac_index=99) is False

    @patch("ouroboros.orchestrator.ac_worktree._branch_exists", return_value=True)
    @patch("ouroboros.orchestrator.ac_worktree._run_git")
    @patch("ouroboros.orchestrator.ac_worktree._ensure_worktree")
    @patch("ouroboros.orchestrator.ac_worktree._run_git_process")
    @patch("ouroboros.orchestrator.ac_worktree._worktree_root")
    def test_remove_all(
        self,
        mock_root: MagicMock,
        mock_git_process: MagicMock,
        mock_ensure: MagicMock,
        mock_run_git: MagicMock,
        mock_branch_exists: MagicMock,
        manager: ACWorktreeManager,
        tmp_path: Path,
    ) -> None:
        mock_root.return_value = tmp_path / "worktrees"
        mock_git_process.return_value = MagicMock(returncode=0)

        manager.create_ac_worktree(ac_index=0)
        manager.create_ac_worktree(ac_index=1)
        manager.create_ac_worktree(ac_index=2)

        removed = manager.remove_all()

        assert removed == 3
        assert manager.active_count == 0


class TestACWorktreeManagerCheckpoint:
    """Test checkpoint serialization/deserialization."""

    @patch("ouroboros.orchestrator.ac_worktree._ensure_worktree")
    @patch("ouroboros.orchestrator.ac_worktree._run_git_process")
    @patch("ouroboros.orchestrator.ac_worktree._worktree_root")
    def test_checkpoint_round_trip(
        self,
        mock_root: MagicMock,
        mock_git_process: MagicMock,
        mock_ensure: MagicMock,
        manager: ACWorktreeManager,
        tmp_path: Path,
    ) -> None:
        mock_root.return_value = tmp_path / "worktrees"
        mock_git_process.return_value = MagicMock(returncode=0)

        manager.create_ac_worktree(ac_index=0)
        manager.create_ac_worktree(ac_index=2)

        checkpoint = manager.to_checkpoint()
        assert len(checkpoint) == 2

        # Create worktree dirs so restore finds them
        for entry in checkpoint:
            Path(entry["worktree_path"]).mkdir(parents=True, exist_ok=True)

        restored = ACWorktreeManager.from_checkpoint(
            execution_id="orch_abc123",
            repo_root=str(tmp_path / "repo"),
            source_cwd=str(tmp_path / "repo"),
            data=checkpoint,
        )

        assert restored.active_count == 2
        assert restored.get_worktree(0) is not None
        assert restored.get_worktree(2) is not None

    def test_checkpoint_skips_missing_directories(self, tmp_path: Path) -> None:
        data = [
            {
                "ac_index": 0,
                "execution_id": "orch_abc123",
                "branch": "ooo/orch_abc123_ac_0",
                "worktree_path": str(tmp_path / "nonexistent"),
                "effective_cwd": str(tmp_path / "nonexistent"),
            },
        ]
        restored = ACWorktreeManager.from_checkpoint(
            execution_id="orch_abc123",
            repo_root=str(tmp_path / "repo"),
            source_cwd=str(tmp_path / "repo"),
            data=data,
        )
        assert restored.active_count == 0


class TestACWorktreeManagerEffectiveCwd:
    """Test effective_cwd calculation with subdir offsets."""

    def test_preserves_subdir_offset(self, tmp_path: Path) -> None:
        repo = tmp_path / "myrepo"
        repo.mkdir()
        source = repo / "src" / "lib"
        source.mkdir(parents=True)

        mgr = ACWorktreeManager(
            execution_id="orch_test",
            repo_root=str(repo),
            source_cwd=str(source),
        )

        wt_path = tmp_path / "worktrees" / "myrepo" / "orch_test_ac_0"
        effective = mgr._effective_cwd(wt_path)
        assert effective == str(wt_path / "src" / "lib")

    def test_falls_back_to_worktree_root_when_outside(self, tmp_path: Path) -> None:
        repo = tmp_path / "myrepo"
        repo.mkdir()
        outside = tmp_path / "elsewhere"
        outside.mkdir()

        mgr = ACWorktreeManager(
            execution_id="orch_test",
            repo_root=str(repo),
            source_cwd=str(outside),
        )

        wt_path = tmp_path / "worktrees" / "myrepo" / "orch_test_ac_0"
        effective = mgr._effective_cwd(wt_path)
        assert effective == str(wt_path)


class TestACWorktreeManagerCommit:
    """Test commit_ac_changes."""

    @patch("ouroboros.orchestrator.ac_worktree._run_git")
    @patch("ouroboros.orchestrator.ac_worktree._ensure_worktree")
    @patch("ouroboros.orchestrator.ac_worktree._run_git_process")
    @patch("ouroboros.orchestrator.ac_worktree._worktree_root")
    def test_commit_returns_sha(
        self,
        mock_root: MagicMock,
        mock_git_process: MagicMock,
        mock_ensure: MagicMock,
        mock_run_git: MagicMock,
        manager: ACWorktreeManager,
        tmp_path: Path,
    ) -> None:
        mock_root.return_value = tmp_path / "worktrees"
        mock_git_process.return_value = MagicMock(returncode=0)

        manager.create_ac_worktree(ac_index=0)

        mock_run_git.reset_mock()
        mock_run_git.side_effect = [
            "M  src/foo.py",  # status --porcelain
            "",  # add -A
            "",  # commit -m
            "abc123def456",  # rev-parse HEAD
        ]

        sha = manager.commit_ac_changes(ac_index=0, message="AC 0 changes")
        assert sha == "abc123def456"

    @patch("ouroboros.orchestrator.ac_worktree._run_git")
    @patch("ouroboros.orchestrator.ac_worktree._ensure_worktree")
    @patch("ouroboros.orchestrator.ac_worktree._run_git_process")
    @patch("ouroboros.orchestrator.ac_worktree._worktree_root")
    def test_commit_nothing_returns_none(
        self,
        mock_root: MagicMock,
        mock_git_process: MagicMock,
        mock_ensure: MagicMock,
        mock_run_git: MagicMock,
        manager: ACWorktreeManager,
        tmp_path: Path,
    ) -> None:
        mock_root.return_value = tmp_path / "worktrees"
        mock_git_process.return_value = MagicMock(returncode=0)

        manager.create_ac_worktree(ac_index=0)

        mock_run_git.reset_mock()
        mock_run_git.return_value = ""  # status --porcelain returns empty

        sha = manager.commit_ac_changes(ac_index=0, message="empty")
        assert sha is None

    def test_commit_untracked_raises(self, manager: ACWorktreeManager) -> None:
        with pytest.raises(WorktreeError, match="No active worktree"):
            manager.commit_ac_changes(ac_index=99, message="nope")


class TestIsolationMode:
    """Test IsolationMode enum."""

    def test_values(self) -> None:
        assert IsolationMode.SHARED.value == "shared"
        assert IsolationMode.WORKTREE.value == "worktree"
