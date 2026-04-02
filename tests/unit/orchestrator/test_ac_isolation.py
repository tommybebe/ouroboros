"""Tests for AC isolation mode classification.

Verifies that:
- ACs with no predicted file overlap get SHARED mode (zero overhead)
- ACs with predicted overlap get WORKTREE mode
- The common case (no overlaps) is truly zero-overhead
- Edge cases are handled correctly
"""

from __future__ import annotations

import pytest

from ouroboros.orchestrator.ac_isolation import (
    ACIsolationPlan,
    IsolationMode,
    classify_isolation_modes,
    needs_worktree,
)


class TestClassifyIsolationModes:
    """Tests for classify_isolation_modes()."""

    def test_no_overlap_all_shared(self) -> None:
        """All ACs get SHARED when no file overlap groups exist."""
        plan = classify_isolation_modes(
            ac_indices=[0, 1, 2],
            file_overlap_groups=None,
        )
        assert plan.all_shared is True
        assert plan.has_worktrees is False
        assert plan.shared_indices == (0, 1, 2)
        assert plan.worktree_indices == ()

    def test_empty_overlap_groups_all_shared(self) -> None:
        """Empty overlap groups list means no overlaps → all shared."""
        plan = classify_isolation_modes(
            ac_indices=[0, 1, 2, 3],
            file_overlap_groups=[],
        )
        assert plan.all_shared is True
        assert plan.shared_indices == (0, 1, 2, 3)

    def test_partial_overlap_mixed_modes(self) -> None:
        """ACs in overlap groups get WORKTREE, others stay SHARED."""
        plan = classify_isolation_modes(
            ac_indices=[0, 1, 2],
            file_overlap_groups=[[0, 1]],
        )
        assert plan.modes[0] == IsolationMode.WORKTREE
        assert plan.modes[1] == IsolationMode.WORKTREE
        assert plan.modes[2] == IsolationMode.SHARED
        assert plan.all_shared is False
        assert plan.has_worktrees is True
        assert plan.shared_indices == (2,)
        assert plan.worktree_indices == (0, 1)

    def test_all_overlap_all_worktree(self) -> None:
        """All ACs overlapping → all get WORKTREE."""
        plan = classify_isolation_modes(
            ac_indices=[0, 1, 2],
            file_overlap_groups=[[0, 1, 2]],
        )
        assert all(m == IsolationMode.WORKTREE for m in plan.modes.values())
        assert plan.all_shared is False
        assert plan.has_worktrees is True

    def test_multiple_overlap_groups(self) -> None:
        """Multiple overlap groups handled correctly."""
        plan = classify_isolation_modes(
            ac_indices=[0, 1, 2, 3, 4],
            file_overlap_groups=[[0, 1], [3, 4]],
        )
        assert plan.modes[0] == IsolationMode.WORKTREE
        assert plan.modes[1] == IsolationMode.WORKTREE
        assert plan.modes[2] == IsolationMode.SHARED
        assert plan.modes[3] == IsolationMode.WORKTREE
        assert plan.modes[4] == IsolationMode.WORKTREE
        assert plan.shared_indices == (2,)
        assert plan.worktree_indices == (0, 1, 3, 4)

    def test_single_ac_group_ignored(self) -> None:
        """Overlap groups with only one AC are ignored (no conflict possible)."""
        plan = classify_isolation_modes(
            ac_indices=[0, 1, 2],
            file_overlap_groups=[[1]],  # Single AC group
        )
        assert plan.all_shared is True
        assert plan.shared_indices == (0, 1, 2)

    def test_single_ac_level(self) -> None:
        """Single AC in a level is always SHARED."""
        plan = classify_isolation_modes(
            ac_indices=[5],
            file_overlap_groups=None,
        )
        assert plan.all_shared is True
        assert plan.modes[5] == IsolationMode.SHARED

    def test_overlap_groups_stored(self) -> None:
        """Overlap groups are normalized and stored in the plan."""
        plan = classify_isolation_modes(
            ac_indices=[0, 1, 2],
            file_overlap_groups=[[0, 2]],
        )
        assert len(plan.overlap_groups) == 1
        assert plan.overlap_groups[0] == frozenset({0, 2})


class TestNeedsWorktree:
    """Tests for the needs_worktree() fast-path gate function."""

    def test_all_shared_plan_returns_false(self) -> None:
        """When plan is all-shared, returns False without dict lookup."""
        plan = classify_isolation_modes(
            ac_indices=[0, 1, 2],
            file_overlap_groups=None,
        )
        assert needs_worktree(plan, 0) is False
        assert needs_worktree(plan, 1) is False
        assert needs_worktree(plan, 2) is False

    def test_worktree_ac_returns_true(self) -> None:
        """AC in overlap group returns True."""
        plan = classify_isolation_modes(
            ac_indices=[0, 1, 2],
            file_overlap_groups=[[0, 1]],
        )
        assert needs_worktree(plan, 0) is True
        assert needs_worktree(plan, 1) is True
        assert needs_worktree(plan, 2) is False

    def test_unknown_ac_returns_false(self) -> None:
        """AC not in plan defaults to SHARED (False)."""
        plan = classify_isolation_modes(
            ac_indices=[0, 1],
            file_overlap_groups=[[0, 1]],
        )
        # AC 99 not in the plan
        assert needs_worktree(plan, 99) is False


class TestACIsolationPlan:
    """Tests for ACIsolationPlan dataclass behavior."""

    def test_frozen(self) -> None:
        """Plan is a frozen dataclass."""
        plan = ACIsolationPlan(modes={0: IsolationMode.SHARED})
        with pytest.raises(AttributeError):
            plan.overlap_groups = ()  # type: ignore[misc]

    def test_to_metadata_serialization(self) -> None:
        """Metadata serialization includes all key fields."""
        plan = classify_isolation_modes(
            ac_indices=[0, 1, 2],
            file_overlap_groups=[[0, 1]],
        )
        meta = plan.to_metadata()
        assert meta["shared_count"] == 1
        assert meta["worktree_count"] == 2
        assert "0" in meta["modes"]
        assert meta["modes"]["0"] == "worktree"
        assert meta["modes"]["2"] == "shared"
        assert len(meta["overlap_groups"]) == 1

    def test_empty_plan(self) -> None:
        """Empty plan is all-shared by default."""
        plan = ACIsolationPlan()
        assert plan.all_shared is True
        assert plan.has_worktrees is False
        assert plan.shared_indices == ()
        assert plan.worktree_indices == ()


class TestZeroOverheadInvariant:
    """Tests specifically verifying the zero-overhead guarantee for the common case.

    The key invariant of AC 3: when no file overlap is predicted, the execution
    path must be identical to today with no additional processing.
    """

    def test_no_overlap_no_worktree_groups_created(self) -> None:
        """No overlap → no overlap_groups created."""
        plan = classify_isolation_modes(
            ac_indices=[0, 1, 2, 3, 4],
            file_overlap_groups=None,
        )
        assert plan.overlap_groups == ()
        assert plan.all_shared is True

    def test_needs_worktree_fast_path(self) -> None:
        """needs_worktree uses all_shared fast path (no dict lookup)."""
        plan = classify_isolation_modes(
            ac_indices=list(range(100)),
            file_overlap_groups=None,
        )
        # all_shared is True → needs_worktree returns False immediately
        assert plan.all_shared is True
        for i in range(100):
            assert needs_worktree(plan, i) is False

    def test_none_overlap_groups_same_as_empty(self) -> None:
        """None and [] produce identical all-shared plans."""
        plan_none = classify_isolation_modes(ac_indices=[0, 1], file_overlap_groups=None)
        plan_empty = classify_isolation_modes(ac_indices=[0, 1], file_overlap_groups=[])
        assert plan_none.all_shared == plan_empty.all_shared
        assert plan_none.shared_indices == plan_empty.shared_indices
        assert plan_none.worktree_indices == plan_empty.worktree_indices
