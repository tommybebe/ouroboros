"""AC isolation mode classification for parallel execution.

Determines whether each AC in a parallel level should execute in the shared
workspace (zero overhead) or in an isolated git worktree (conflict prevention).

The key invariant: ACs with no predicted file overlap use IsolationMode.SHARED
and follow the exact same execution path as today — no worktree setup, no
merge step, no additional overhead whatsoever.

Usage:
    from ouroboros.orchestrator.ac_isolation import (
        classify_isolation_modes,
        IsolationMode,
    )

    modes = classify_isolation_modes(
        ac_indices=[0, 1, 2],
        file_overlap_groups=[[0, 1]],  # AC 0 and 1 overlap
    )
    # modes == {0: WORKTREE, 1: WORKTREE, 2: SHARED}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ouroboros.observability.logging import get_logger

log = get_logger(__name__)


class IsolationMode(Enum):
    """Execution isolation strategy for a single AC.

    SHARED:   Run in the shared task workspace (current behavior, zero overhead).
    WORKTREE: Run in a dedicated git worktree for conflict-safe isolation.
    """

    SHARED = "shared"
    WORKTREE = "worktree"


@dataclass(frozen=True, slots=True)
class ACIsolationPlan:
    """Isolation decisions for all ACs in a parallel execution level.

    Attributes:
        modes: Mapping of AC index to its isolation mode.
        overlap_groups: Groups of AC indices that share predicted file overlap.
            Each group is a frozenset of AC indices whose predicted files
            intersect. ACs not in any group have no predicted overlap.
    """

    modes: dict[int, IsolationMode] = field(default_factory=dict)
    overlap_groups: tuple[frozenset[int], ...] = field(default_factory=tuple)

    @property
    def shared_indices(self) -> tuple[int, ...]:
        """AC indices that will run in the shared workspace."""
        return tuple(
            sorted(idx for idx, mode in self.modes.items() if mode == IsolationMode.SHARED)
        )

    @property
    def worktree_indices(self) -> tuple[int, ...]:
        """AC indices that will run in isolated worktrees."""
        return tuple(
            sorted(idx for idx, mode in self.modes.items() if mode == IsolationMode.WORKTREE)
        )

    @property
    def all_shared(self) -> bool:
        """True when every AC uses the shared workspace (common case)."""
        return all(mode == IsolationMode.SHARED for mode in self.modes.values())

    @property
    def has_worktrees(self) -> bool:
        """True when at least one AC requires worktree isolation."""
        return any(mode == IsolationMode.WORKTREE for mode in self.modes.values())

    def to_metadata(self) -> dict[str, Any]:
        """Serialize for event/checkpoint storage."""
        return {
            "modes": {str(k): v.value for k, v in self.modes.items()},
            "overlap_groups": [sorted(g) for g in self.overlap_groups],
            "shared_count": len(self.shared_indices),
            "worktree_count": len(self.worktree_indices),
        }


def classify_isolation_modes(
    ac_indices: list[int] | tuple[int, ...],
    file_overlap_groups: list[list[int]] | None = None,
) -> ACIsolationPlan:
    """Classify each AC's isolation mode based on predicted file overlap.

    Conservative strategy: any AC that appears in a file overlap group gets
    WORKTREE isolation. ACs with no overlap get SHARED (zero overhead).

    When ``file_overlap_groups`` is None or empty, ALL ACs get SHARED mode —
    this is the common case and the zero-overhead fast path.

    Args:
        ac_indices: All AC indices in this parallel level.
        file_overlap_groups: Groups of AC indices with predicted file overlap.
            Each group is a list of AC indices whose predicted target files
            intersect. Produced by the DependencyAnalyzer's file overlap
            prediction at planning time.

    Returns:
        ACIsolationPlan with per-AC isolation decisions.
    """
    modes: dict[int, IsolationMode] = {}

    # Fast path: no overlap predictions → all shared (zero overhead)
    if not file_overlap_groups:
        for idx in ac_indices:
            modes[idx] = IsolationMode.SHARED
        log.debug(
            "ac_isolation.all_shared",
            ac_count=len(ac_indices),
            reason="no_file_overlap_predicted",
        )
        return ACIsolationPlan(modes=modes, overlap_groups=())

    # Collect all AC indices that participate in any overlap group
    overlapping_indices: set[int] = set()
    normalized_groups: list[frozenset[int]] = []
    for group in file_overlap_groups:
        if len(group) >= 2:
            frozen = frozenset(group)
            normalized_groups.append(frozen)
            overlapping_indices.update(frozen)

    # Assign modes: overlapping → WORKTREE, non-overlapping → SHARED
    for idx in ac_indices:
        if idx in overlapping_indices:
            modes[idx] = IsolationMode.WORKTREE
        else:
            modes[idx] = IsolationMode.SHARED

    shared_count = sum(1 for m in modes.values() if m == IsolationMode.SHARED)
    worktree_count = sum(1 for m in modes.values() if m == IsolationMode.WORKTREE)

    log.info(
        "ac_isolation.classified",
        ac_count=len(ac_indices),
        shared_count=shared_count,
        worktree_count=worktree_count,
        overlap_groups=len(normalized_groups),
    )

    return ACIsolationPlan(
        modes=modes,
        overlap_groups=tuple(normalized_groups),
    )


def needs_worktree(plan: ACIsolationPlan, ac_index: int) -> bool:
    """Check whether a specific AC requires worktree isolation.

    Returns False (shared workspace) when:
    - The AC is not in the plan (default: shared)
    - The AC's mode is SHARED
    - The plan has no worktree assignments at all

    This is the gate function called in the hot path of AC dispatch.
    For the common case (no overlaps), ``plan.all_shared`` is True and
    this function returns False immediately without dict lookup overhead.
    """
    if plan.all_shared:
        return False
    return plan.modes.get(ac_index, IsolationMode.SHARED) == IsolationMode.WORKTREE


__all__ = [
    "ACIsolationPlan",
    "IsolationMode",
    "classify_isolation_modes",
    "needs_worktree",
]
