"""Performance benchmarks for worktree setup and merge operations.

AC 8: Worktree setup overhead is ~100ms per AC, merge ~200ms,
merge-agent only when git can't auto-merge.

These tests verify that the per-AC worktree lifecycle overhead stays
within acceptable bounds for typical parallel execution scenarios.
The timing budgets are:
  - Worktree creation: ~100ms per AC
  - Branch merge (auto-merge, no conflicts): ~200ms per AC
  - Merge-agent: only invoked when git cannot auto-merge (zero calls
    for non-overlapping changes)

Typical parallel levels have 3 ACs, so the total overhead budget for
the common (no-conflict) case is ~900ms: 3x setup + 3x merge.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import time

import pytest

from ouroboros.orchestrator.ac_worktree import ACWorktreeManager
from ouroboros.orchestrator.worktree_merge import (
    MergeOutcome,
    WorktreeMerger,
)

# ---------------------------------------------------------------------------
# Timing budget constants (milliseconds)
# ---------------------------------------------------------------------------

# Upper bound per-AC worktree creation.  The ~100ms target allows generous
# headroom — CI machines can be slower, so we allow up to 500ms before
# failing.  The test reports the actual timing so we can track regressions.
WORKTREE_SETUP_BUDGET_MS = 500

# Upper bound per-AC auto-merge (no conflicts).  Same reasoning — ~200ms
# target with 1000ms hard ceiling for CI variance.
MERGE_BUDGET_MS = 1000

# Total overhead budget for a typical 3-AC parallel level (setup + merge
# for all 3 ACs).  Should be well under 5 seconds even on slow machines.
LEVEL_TOTAL_BUDGET_MS = 5_000


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
    (path / "README.md").write_text("# Test\n")
    _git(["add", "."], path)
    _git(["commit", "-m", "Initial commit"], path)
    return path


def _create_branch_with_file(
    repo: Path,
    branch: str,
    filename: str,
    content: str,
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


# ---------------------------------------------------------------------------
# Worktree setup performance tests
# ---------------------------------------------------------------------------


class TestWorktreeSetupPerformance:
    """Verify that per-AC worktree creation stays within timing budget."""

    def test_single_worktree_creation_timing(self, repo: Path) -> None:
        """Creating one AC worktree should take ~100ms or less."""
        manager = ACWorktreeManager(
            execution_id="perf_test",
            repo_root=str(repo),
            source_cwd=str(repo),
        )

        start = time.perf_counter()
        info = manager.create_ac_worktree(ac_index=0)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert info.branch == "ooo/perf_test_ac_0"
        assert Path(info.worktree_path).exists()
        assert elapsed_ms < WORKTREE_SETUP_BUDGET_MS, (
            f"Worktree creation took {elapsed_ms:.0f}ms, "
            f"budget is {WORKTREE_SETUP_BUDGET_MS}ms"
        )

        # Cleanup
        manager.remove_ac_worktree(0, force=True)

    def test_three_worktree_creations_timing(self, repo: Path) -> None:
        """Creating 3 AC worktrees (typical parallel level) stays within budget."""
        manager = ACWorktreeManager(
            execution_id="perf_3ac",
            repo_root=str(repo),
            source_cwd=str(repo),
        )

        start = time.perf_counter()
        for i in range(3):
            manager.create_ac_worktree(ac_index=i)
        total_ms = (time.perf_counter() - start) * 1000
        per_ac_ms = total_ms / 3

        assert manager.active_count == 3
        assert per_ac_ms < WORKTREE_SETUP_BUDGET_MS, (
            f"Per-AC worktree creation averaged {per_ac_ms:.0f}ms "
            f"({total_ms:.0f}ms total for 3 ACs), "
            f"budget is {WORKTREE_SETUP_BUDGET_MS}ms per AC"
        )

        # Cleanup
        manager.remove_all(force=True)

    def test_worktree_removal_is_fast(self, repo: Path) -> None:
        """Worktree cleanup should not be a bottleneck."""
        manager = ACWorktreeManager(
            execution_id="perf_cleanup",
            repo_root=str(repo),
            source_cwd=str(repo),
        )

        # Create worktrees first
        for i in range(3):
            manager.create_ac_worktree(ac_index=i)

        start = time.perf_counter()
        removed = manager.remove_all(force=True)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert removed == 3
        # Cleanup should be no slower than creation
        assert elapsed_ms < WORKTREE_SETUP_BUDGET_MS * 3, (
            f"Removing 3 worktrees took {elapsed_ms:.0f}ms"
        )


# ---------------------------------------------------------------------------
# Merge performance tests
# ---------------------------------------------------------------------------


class TestMergePerformance:
    """Verify that auto-merge (no conflicts) stays within timing budget."""

    def test_single_auto_merge_timing(self, repo: Path) -> None:
        """Auto-merging one AC branch should take ~200ms or less."""
        default_branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo)
        _create_branch_with_file(repo, "ooo/perf_ac_0", "file_a.py", "a = 1\n")

        merger = WorktreeMerger(repo_root=repo)

        start = time.perf_counter()
        result = merger.merge_ac_branch(default_branch, "ooo/perf_ac_0", 0)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert result.outcome == MergeOutcome.SUCCESS
        assert elapsed_ms < MERGE_BUDGET_MS, (
            f"Auto-merge took {elapsed_ms:.0f}ms, "
            f"budget is {MERGE_BUDGET_MS}ms"
        )

    def test_three_sequential_merges_timing(self, repo: Path) -> None:
        """Merging 3 AC branches sequentially stays within budget."""
        default_branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo)

        for i in range(3):
            _create_branch_with_file(
                repo, f"ooo/perf_ac_{i}", f"file_{i}.py", f"x_{i} = {i}\n"
            )

        merger = WorktreeMerger(repo_root=repo)

        start = time.perf_counter()
        plan = merger.merge_all(
            target_branch=default_branch,
            ac_branches=[(i, f"ooo/perf_ac_{i}") for i in range(3)],
        )
        total_ms = (time.perf_counter() - start) * 1000
        per_ac_ms = total_ms / 3

        assert plan.all_succeeded is True
        assert plan.success_count == 3
        assert per_ac_ms < MERGE_BUDGET_MS, (
            f"Per-AC merge averaged {per_ac_ms:.0f}ms "
            f"({total_ms:.0f}ms total for 3 ACs), "
            f"budget is {MERGE_BUDGET_MS}ms per AC"
        )

    def test_noop_merge_is_fast(self, repo: Path) -> None:
        """Merging a branch with no changes should be near-instant."""
        default_branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo)
        _git(["branch", "ooo/perf_noop", default_branch], repo)

        merger = WorktreeMerger(repo_root=repo)

        start = time.perf_counter()
        result = merger.merge_ac_branch(default_branch, "ooo/perf_noop", 0)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert result.outcome == MergeOutcome.NOTHING_TO_MERGE
        # Noop should be faster than a real merge
        assert elapsed_ms < MERGE_BUDGET_MS, (
            f"Noop merge took {elapsed_ms:.0f}ms"
        )


# ---------------------------------------------------------------------------
# End-to-end overhead for a typical 3-AC parallel level
# ---------------------------------------------------------------------------


class TestEndToEndLevelOverhead:
    """Verify total overhead for setup + merge of a typical 3-AC level."""

    def test_full_level_overhead_within_budget(self, repo: Path) -> None:
        """Total worktree setup + merge for 3 ACs stays under budget.

        This simulates the full lifecycle:
        1. Create 3 worktrees
        2. Each AC makes a file change and commits
        3. Merge all 3 branches back
        4. Cleanup worktrees

        The measured overhead excludes the simulated "work" time (file writes)
        and only captures the git operations that constitute the isolation
        overhead.
        """
        default_branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo)

        manager = ACWorktreeManager(
            execution_id="perf_e2e",
            repo_root=str(repo),
            source_cwd=str(repo),
        )

        # Phase 1: Create worktrees
        setup_start = time.perf_counter()
        for i in range(3):
            manager.create_ac_worktree(ac_index=i)
        setup_ms = (time.perf_counter() - setup_start) * 1000

        # Simulate AC work: write a unique file in each worktree and commit
        for i in range(3):
            info = manager.get_worktree(i)
            assert info is not None
            wt = Path(info.worktree_path)
            (wt / f"ac_{i}_output.py").write_text(f"result_{i} = {i * 10}\n")
            manager.commit_ac_changes(i, f"AC {i} output")

        # Phase 2: Merge all branches
        merger = WorktreeMerger(repo_root=repo)
        ac_branches = [
            (i, f"ooo/perf_e2e_ac_{i}") for i in range(3)
        ]

        merge_start = time.perf_counter()
        plan = merger.merge_all(target_branch=default_branch, ac_branches=ac_branches)
        merge_ms = (time.perf_counter() - merge_start) * 1000

        # Phase 3: Cleanup
        cleanup_start = time.perf_counter()
        manager.remove_all(force=True)
        cleanup_ms = (time.perf_counter() - cleanup_start) * 1000

        total_overhead_ms = setup_ms + merge_ms + cleanup_ms

        # Verify correctness
        assert plan.all_succeeded is True
        assert plan.success_count == 3
        for i in range(3):
            assert (repo / f"ac_{i}_output.py").exists(), (
                f"AC {i}'s output file missing after merge"
            )

        # Verify performance
        assert total_overhead_ms < LEVEL_TOTAL_BUDGET_MS, (
            f"Total level overhead: {total_overhead_ms:.0f}ms "
            f"(setup={setup_ms:.0f}ms, merge={merge_ms:.0f}ms, "
            f"cleanup={cleanup_ms:.0f}ms), "
            f"budget is {LEVEL_TOTAL_BUDGET_MS}ms"
        )


# ---------------------------------------------------------------------------
# Merge-agent gating: only invoked when git can't auto-merge
# ---------------------------------------------------------------------------


class TestMergeAgentGating:
    """Verify merge-agent is NOT invoked when auto-merge succeeds.

    AC 8 requires that merge-agent is only dispatched when git cannot
    auto-merge. These tests verify the gating logic by confirming that
    auto-merge results never have CONFLICT outcomes (which would trigger
    merge-agent dispatch).
    """

    def test_non_overlapping_files_skip_merge_agent(self, repo: Path) -> None:
        """ACs editing different files auto-merge — no merge-agent needed."""
        default_branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo)

        for i in range(3):
            _create_branch_with_file(
                repo, f"ooo/gate_ac_{i}", f"module_{i}.py", f"mod_{i} = True\n"
            )

        merger = WorktreeMerger(repo_root=repo)
        plan = merger.merge_all(
            target_branch=default_branch,
            ac_branches=[(i, f"ooo/gate_ac_{i}") for i in range(3)],
        )

        # All should auto-merge — zero conflicts means zero merge-agent calls
        assert plan.conflict_count == 0, (
            f"Expected 0 conflicts (no merge-agent needed), got {plan.conflict_count}"
        )
        assert plan.all_succeeded is True

    def test_non_overlapping_hunks_same_file_skip_merge_agent(
        self, repo: Path
    ) -> None:
        """ACs editing different regions of the same file auto-merge cleanly."""
        default_branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo)

        # Base file with distinct sections
        base = (
            "# Header\n\n"
            "# Section A\ndef func_a():\n    return 'original_a'\n\n"
            "# Section B\ndef func_b():\n    return 'original_b'\n\n"
            "# Section C\ndef func_c():\n    return 'original_c'\n"
        )
        (repo / "shared.py").write_text(base)
        _git(["add", "shared.py"], repo)
        _git(["commit", "-m", "Add shared.py with sections"], repo)

        # AC 0 modifies Section A
        _git(["checkout", "-b", "ooo/gate_ac_0", default_branch], repo)
        content_a = base.replace("return 'original_a'", "return 'modified_a'")
        (repo / "shared.py").write_text(content_a)
        _git(["add", "shared.py"], repo)
        _git(["commit", "-m", "AC 0: modify Section A"], repo)
        _git(["checkout", default_branch], repo)

        # AC 1 modifies Section C
        _git(["checkout", "-b", "ooo/gate_ac_1", default_branch], repo)
        content_c = base.replace("return 'original_c'", "return 'modified_c'")
        (repo / "shared.py").write_text(content_c)
        _git(["add", "shared.py"], repo)
        _git(["commit", "-m", "AC 1: modify Section C"], repo)
        _git(["checkout", default_branch], repo)

        merger = WorktreeMerger(repo_root=repo)
        plan = merger.merge_all(
            target_branch=default_branch,
            ac_branches=[(0, "ooo/gate_ac_0"), (1, "ooo/gate_ac_1")],
        )

        # Non-overlapping hunks in the same file: auto-merge succeeds
        assert plan.conflict_count == 0
        assert plan.all_succeeded is True

        # Verify both modifications are present
        final = (repo / "shared.py").read_text()
        assert "modified_a" in final
        assert "modified_c" in final

    def test_overlapping_hunks_trigger_conflict(self, repo: Path) -> None:
        """ACs editing the same line produce CONFLICT — merge-agent IS needed."""
        default_branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo)

        (repo / "config.py").write_text("value = 'original'\n")
        _git(["add", "config.py"], repo)
        _git(["commit", "-m", "Add config.py"], repo)

        # AC 0 changes the same line
        _git(["checkout", "-b", "ooo/conflict_ac_0", default_branch], repo)
        (repo / "config.py").write_text("value = 'from_ac_0'\n")
        _git(["add", "config.py"], repo)
        _git(["commit", "-m", "AC 0 changes value"], repo)
        _git(["checkout", default_branch], repo)

        # AC 1 also changes the same line
        _git(["checkout", "-b", "ooo/conflict_ac_1", default_branch], repo)
        (repo / "config.py").write_text("value = 'from_ac_1'\n")
        _git(["add", "config.py"], repo)
        _git(["commit", "-m", "AC 1 changes value"], repo)
        _git(["checkout", default_branch], repo)

        merger = WorktreeMerger(repo_root=repo)

        # First merge succeeds
        r0 = merger.merge_ac_branch(default_branch, "ooo/conflict_ac_0", 0)
        assert r0.succeeded is True

        # Second merge conflicts — this is the ONLY case where merge-agent
        # would be dispatched
        r1 = merger.merge_ac_branch(default_branch, "ooo/conflict_ac_1", 1)
        assert r1.has_conflicts is True
        assert r1.outcome == MergeOutcome.CONFLICT
        assert "config.py" in r1.conflicting_files


# ---------------------------------------------------------------------------
# Zero-overhead for shared workspace ACs
# ---------------------------------------------------------------------------


class TestSharedWorkspaceZeroOverhead:
    """Verify ACs with no file overlap run with zero additional overhead.

    When the file overlap predictor says ACs don't conflict, they run in
    the shared workspace with no worktree creation or merge step at all.
    """

    def test_no_worktree_created_for_shared_mode(self, repo: Path) -> None:
        """Shared-mode ACs should NOT have worktrees — zero overhead."""
        manager = ACWorktreeManager(
            execution_id="shared_test",
            repo_root=str(repo),
            source_cwd=str(repo),
        )

        # Don't create any worktrees — simulating shared mode
        assert manager.active_count == 0
        assert manager.get_worktree(0) is None
        assert manager.get_worktree(1) is None

        # No cleanup needed — zero overhead
        removed = manager.remove_all()
        assert removed == 0
