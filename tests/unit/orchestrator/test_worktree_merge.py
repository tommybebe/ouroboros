"""Tests for worktree merge operations.

Focuses on AC 4: When git auto-merge succeeds (no overlapping hunks),
merge completes without agent intervention.
"""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from ouroboros.orchestrator.worktree_merge import (
    MergeOutcome,
    MergePlanResult,
    MergeResult,
    WorktreeMerger,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git(args: list[str], cwd: Path) -> str:
    """Run a git command and return stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _init_repo(path: Path) -> Path:
    """Create a minimal git repo with an initial commit."""
    path.mkdir(parents=True, exist_ok=True)
    _git(["init"], path)
    _git(["config", "user.email", "test@test.com"], path)
    _git(["config", "user.name", "Test"], path)
    # Initial commit
    (path / "README.md").write_text("# Test\n")
    _git(["add", "."], path)
    _git(["commit", "-m", "Initial commit"], path)
    return path


def _create_branch_with_file(
    repo: Path,
    branch: str,
    filename: str,
    content: str,
    *,
    base_branch: str = "main",
) -> str:
    """Create a branch with a single file change and return the commit SHA."""
    default_branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo)
    _git(["checkout", "-b", branch, default_branch], repo)
    (repo / filename).write_text(content)
    _git(["add", filename], repo)
    _git(["commit", "-m", f"Add {filename} on {branch}"], repo)
    sha = _git(["rev-parse", "HEAD"], repo)
    _git(["checkout", default_branch], repo)
    return sha


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """Create a test git repo."""
    return _init_repo(tmp_path / "repo")


@pytest.fixture()
def merger(repo: Path) -> WorktreeMerger:
    """Create a WorktreeMerger for the test repo."""
    return WorktreeMerger(repo_root=repo)


# ---------------------------------------------------------------------------
# MergeResult dataclass tests
# ---------------------------------------------------------------------------


class TestMergeResult:
    """Test MergeResult properties and serialization."""

    def test_succeeded_property(self) -> None:
        result = MergeResult(
            ac_index=0,
            ac_branch="ooo/test_ac_0",
            outcome=MergeOutcome.SUCCESS,
            merge_sha="abc123",
        )
        assert result.succeeded is True
        assert result.has_conflicts is False
        assert result.is_noop is False

    def test_conflict_property(self) -> None:
        result = MergeResult(
            ac_index=1,
            ac_branch="ooo/test_ac_1",
            outcome=MergeOutcome.CONFLICT,
            conflicting_files=("src/main.py",),
        )
        assert result.succeeded is False
        assert result.has_conflicts is True
        assert result.is_noop is False

    def test_noop_property(self) -> None:
        result = MergeResult(
            ac_index=2,
            ac_branch="ooo/test_ac_2",
            outcome=MergeOutcome.NOTHING_TO_MERGE,
        )
        assert result.succeeded is False
        assert result.has_conflicts is False
        assert result.is_noop is True

    def test_to_dict_roundtrip(self) -> None:
        result = MergeResult(
            ac_index=0,
            ac_branch="ooo/test_ac_0",
            outcome=MergeOutcome.SUCCESS,
            merge_sha="abc123def",
            warnings=("Warning 1",),
        )
        d = result.to_dict()
        assert d["ac_index"] == 0
        assert d["ac_branch"] == "ooo/test_ac_0"
        assert d["outcome"] == "success"
        assert d["merge_sha"] == "abc123def"
        assert d["warnings"] == ["Warning 1"]


class TestMergePlanResult:
    """Test MergePlanResult aggregation properties."""

    def test_all_succeeded_when_all_success(self) -> None:
        plan = MergePlanResult(
            results=(
                MergeResult(ac_index=0, ac_branch="b0", outcome=MergeOutcome.SUCCESS),
                MergeResult(ac_index=1, ac_branch="b1", outcome=MergeOutcome.SUCCESS),
            )
        )
        assert plan.all_succeeded is True
        assert plan.success_count == 2
        assert plan.conflict_count == 0

    def test_all_succeeded_includes_noop(self) -> None:
        plan = MergePlanResult(
            results=(
                MergeResult(ac_index=0, ac_branch="b0", outcome=MergeOutcome.SUCCESS),
                MergeResult(ac_index=1, ac_branch="b1", outcome=MergeOutcome.NOTHING_TO_MERGE),
            )
        )
        assert plan.all_succeeded is True

    def test_not_all_succeeded_with_conflict(self) -> None:
        plan = MergePlanResult(
            results=(
                MergeResult(ac_index=0, ac_branch="b0", outcome=MergeOutcome.SUCCESS),
                MergeResult(ac_index=1, ac_branch="b1", outcome=MergeOutcome.CONFLICT),
            )
        )
        assert plan.all_succeeded is False
        assert plan.conflict_count == 1

    def test_conflict_results_filter(self) -> None:
        plan = MergePlanResult(
            results=(
                MergeResult(ac_index=0, ac_branch="b0", outcome=MergeOutcome.SUCCESS),
                MergeResult(ac_index=1, ac_branch="b1", outcome=MergeOutcome.CONFLICT),
                MergeResult(ac_index=2, ac_branch="b2", outcome=MergeOutcome.CONFLICT),
            )
        )
        conflicts = plan.conflict_results
        assert len(conflicts) == 2
        assert conflicts[0].ac_index == 1
        assert conflicts[1].ac_index == 2

    def test_warnings_for_next_level_aggregation(self) -> None:
        plan = MergePlanResult(
            results=(
                MergeResult(
                    ac_index=0, ac_branch="b0", outcome=MergeOutcome.SUCCESS,
                    warnings=("w1",),
                ),
                MergeResult(
                    ac_index=1, ac_branch="b1", outcome=MergeOutcome.SUCCESS,
                    warnings=("w2", "w3"),
                ),
            )
        )
        assert plan.warnings_for_next_level == ("w1", "w2", "w3")

    def test_empty_plan(self) -> None:
        plan = MergePlanResult()
        assert plan.all_succeeded is True
        assert plan.success_count == 0
        assert plan.conflict_count == 0


# ---------------------------------------------------------------------------
# Integration tests with real git repos
# ---------------------------------------------------------------------------


class TestAutoMergeSuccess:
    """AC 4: When git auto-merge succeeds, merge completes without agent intervention."""

    def test_merge_non_overlapping_files_succeeds(self, repo: Path, merger: WorktreeMerger) -> None:
        """Two ACs editing different files auto-merge cleanly."""
        default_branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo)

        # AC 0 adds file_a.py
        _create_branch_with_file(repo, "ooo/test_ac_0", "file_a.py", "print('a')\n")
        # AC 1 adds file_b.py
        _create_branch_with_file(repo, "ooo/test_ac_1", "file_b.py", "print('b')\n")

        # Merge AC 0
        r0 = merger.merge_ac_branch(default_branch, "ooo/test_ac_0", 0)
        assert r0.outcome == MergeOutcome.SUCCESS
        assert r0.succeeded is True
        assert r0.merge_sha is not None
        assert len(r0.merge_sha) == 40  # full SHA

        # Merge AC 1 (on top of AC 0's merge)
        r1 = merger.merge_ac_branch(default_branch, "ooo/test_ac_1", 1)
        assert r1.outcome == MergeOutcome.SUCCESS
        assert r1.succeeded is True
        assert r1.merge_sha is not None

        # Verify both files are present
        assert (repo / "file_a.py").exists()
        assert (repo / "file_b.py").exists()

    def test_merge_same_file_non_overlapping_hunks_succeeds(
        self, repo: Path, merger: WorktreeMerger
    ) -> None:
        """Two ACs editing different parts of the same file auto-merge cleanly."""
        default_branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo)

        # Create a file with distinct sections
        base_content = "# Header\n\n# Section A\noriginal_a\n\n# Section B\noriginal_b\n"
        (repo / "shared.py").write_text(base_content)
        _git(["add", "shared.py"], repo)
        _git(["commit", "-m", "Add shared.py"], repo)

        # AC 0 modifies Section A
        _git(["checkout", "-b", "ooo/test_ac_0", default_branch], repo)
        content_a = "# Header\n\n# Section A\nmodified_by_ac_0\n\n# Section B\noriginal_b\n"
        (repo / "shared.py").write_text(content_a)
        _git(["add", "shared.py"], repo)
        _git(["commit", "-m", "AC 0 modifies Section A"], repo)
        _git(["checkout", default_branch], repo)

        # AC 1 modifies Section B
        _git(["checkout", "-b", "ooo/test_ac_1", default_branch], repo)
        content_b = "# Header\n\n# Section A\noriginal_a\n\n# Section B\nmodified_by_ac_1\n"
        (repo / "shared.py").write_text(content_b)
        _git(["add", "shared.py"], repo)
        _git(["commit", "-m", "AC 1 modifies Section B"], repo)
        _git(["checkout", default_branch], repo)

        # Both should auto-merge without conflicts
        r0 = merger.merge_ac_branch(default_branch, "ooo/test_ac_0", 0)
        assert r0.succeeded is True

        r1 = merger.merge_ac_branch(default_branch, "ooo/test_ac_1", 1)
        assert r1.succeeded is True

        # Verify both changes are present
        final_content = (repo / "shared.py").read_text()
        assert "modified_by_ac_0" in final_content
        assert "modified_by_ac_1" in final_content

    def test_nothing_to_merge_when_no_changes(self, repo: Path, merger: WorktreeMerger) -> None:
        """AC branch with no changes returns NOTHING_TO_MERGE."""
        default_branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo)

        # Create branch with no changes
        _git(["branch", "ooo/test_ac_0", default_branch], repo)

        result = merger.merge_ac_branch(default_branch, "ooo/test_ac_0", 0)
        assert result.outcome == MergeOutcome.NOTHING_TO_MERGE
        assert result.is_noop is True

    def test_merge_all_non_overlapping(self, repo: Path, merger: WorktreeMerger) -> None:
        """merge_all with non-overlapping changes succeeds for all ACs."""
        default_branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo)

        _create_branch_with_file(repo, "ooo/test_ac_0", "a.py", "a\n")
        _create_branch_with_file(repo, "ooo/test_ac_1", "b.py", "b\n")
        _create_branch_with_file(repo, "ooo/test_ac_2", "c.py", "c\n")

        plan = merger.merge_all(
            target_branch=default_branch,
            ac_branches=[
                (0, "ooo/test_ac_0"),
                (1, "ooo/test_ac_1"),
                (2, "ooo/test_ac_2"),
            ],
        )

        assert plan.all_succeeded is True
        assert plan.success_count == 3
        assert plan.conflict_count == 0
        assert len(plan.conflict_results) == 0

        # Verify all files present
        assert (repo / "a.py").exists()
        assert (repo / "b.py").exists()
        assert (repo / "c.py").exists()

    def test_merge_all_sorts_by_ac_index(self, repo: Path, merger: WorktreeMerger) -> None:
        """merge_all processes branches in ac_index order regardless of input order."""
        default_branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo)

        _create_branch_with_file(repo, "ooo/test_ac_2", "c.py", "c\n")
        _create_branch_with_file(repo, "ooo/test_ac_0", "a.py", "a\n")
        _create_branch_with_file(repo, "ooo/test_ac_1", "b.py", "b\n")

        # Pass in unsorted order
        plan = merger.merge_all(
            target_branch=default_branch,
            ac_branches=[
                (2, "ooo/test_ac_2"),
                (0, "ooo/test_ac_0"),
                (1, "ooo/test_ac_1"),
            ],
        )

        assert plan.all_succeeded is True
        # Verify order of results
        assert plan.results[0].ac_index == 0
        assert plan.results[1].ac_index == 1
        assert plan.results[2].ac_index == 2


class TestConflictDetection:
    """Verify conflict path returns proper MergeResult (complements AC 5)."""

    def test_overlapping_changes_detected_as_conflict(
        self, repo: Path, merger: WorktreeMerger
    ) -> None:
        """Two ACs editing the same line produce a CONFLICT result."""
        default_branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo)

        # Create a base file
        (repo / "config.py").write_text("value = 'original'\n")
        _git(["add", "config.py"], repo)
        _git(["commit", "-m", "Add config.py"], repo)

        # AC 0 changes the value
        _git(["checkout", "-b", "ooo/test_ac_0", default_branch], repo)
        (repo / "config.py").write_text("value = 'from_ac_0'\n")
        _git(["add", "config.py"], repo)
        _git(["commit", "-m", "AC 0 changes value"], repo)
        _git(["checkout", default_branch], repo)

        # AC 1 also changes the same value
        _git(["checkout", "-b", "ooo/test_ac_1", default_branch], repo)
        (repo / "config.py").write_text("value = 'from_ac_1'\n")
        _git(["add", "config.py"], repo)
        _git(["commit", "-m", "AC 1 changes value"], repo)
        _git(["checkout", default_branch], repo)

        # First merge succeeds
        r0 = merger.merge_ac_branch(default_branch, "ooo/test_ac_0", 0)
        assert r0.succeeded is True

        # Second merge should conflict
        r1 = merger.merge_ac_branch(default_branch, "ooo/test_ac_1", 1)
        assert r1.has_conflicts is True
        assert r1.outcome == MergeOutcome.CONFLICT
        assert "config.py" in r1.conflicting_files
        assert r1.conflict_diff  # Non-empty diff

        # Verify repo is left in clean state (merge aborted)
        status = _git(["status", "--porcelain"], repo)
        assert status == ""

    def test_merge_all_with_partial_conflicts(
        self, repo: Path, merger: WorktreeMerger
    ) -> None:
        """merge_all continues past conflicts and reports all results."""
        default_branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo)

        # Base file
        (repo / "data.py").write_text("x = 1\n")
        _git(["add", "data.py"], repo)
        _git(["commit", "-m", "Add data.py"], repo)

        # AC 0: changes data.py
        _git(["checkout", "-b", "ooo/test_ac_0", default_branch], repo)
        (repo / "data.py").write_text("x = 100\n")
        _git(["add", "data.py"], repo)
        _git(["commit", "-m", "AC 0 edits data"], repo)
        _git(["checkout", default_branch], repo)

        # AC 1: also changes data.py (will conflict with AC 0)
        _git(["checkout", "-b", "ooo/test_ac_1", default_branch], repo)
        (repo / "data.py").write_text("x = 200\n")
        _git(["add", "data.py"], repo)
        _git(["commit", "-m", "AC 1 edits data"], repo)
        _git(["checkout", default_branch], repo)

        # AC 2: adds a new file (no conflict)
        _create_branch_with_file(repo, "ooo/test_ac_2", "other.py", "y = 3\n")

        plan = merger.merge_all(
            target_branch=default_branch,
            ac_branches=[
                (0, "ooo/test_ac_0"),
                (1, "ooo/test_ac_1"),
                (2, "ooo/test_ac_2"),
            ],
        )

        assert plan.all_succeeded is False
        assert plan.success_count == 2  # AC 0 and AC 2
        assert plan.conflict_count == 1  # AC 1
        assert plan.conflict_results[0].ac_index == 1


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_error_on_nonexistent_branch(self, repo: Path, merger: WorktreeMerger) -> None:
        """Merging a nonexistent branch returns ERROR."""
        default_branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo)
        result = merger.merge_ac_branch(default_branch, "ooo/nonexistent", 0)
        # Could be ERROR or NOTHING_TO_MERGE depending on git behavior
        assert result.outcome in (MergeOutcome.ERROR, MergeOutcome.NOTHING_TO_MERGE)

    def test_merge_result_to_dict_preserves_fields(self) -> None:
        """Verify serialization captures all relevant fields."""
        result = MergeResult(
            ac_index=3,
            ac_branch="ooo/test_ac_3",
            outcome=MergeOutcome.CONFLICT,
            conflicting_files=("a.py", "b.py"),
            error_message="",
        )
        d = result.to_dict()
        assert d["outcome"] == "conflict"
        assert d["conflicting_files"] == ["a.py", "b.py"]
        assert d["ac_index"] == 3

    def test_plan_to_dict(self) -> None:
        """Verify MergePlanResult serialization."""
        plan = MergePlanResult(
            results=(
                MergeResult(ac_index=0, ac_branch="b0", outcome=MergeOutcome.SUCCESS,
                            merge_sha="abc"),
                MergeResult(ac_index=1, ac_branch="b1", outcome=MergeOutcome.CONFLICT),
            )
        )
        d = plan.to_dict()
        assert d["all_succeeded"] is False
        assert d["success_count"] == 1
        assert d["conflict_count"] == 1
        assert len(d["results"]) == 2
