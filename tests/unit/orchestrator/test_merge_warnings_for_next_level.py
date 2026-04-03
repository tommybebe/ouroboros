"""Tests for merge-agent warnings_for_next_level flow (AC 6).

Verifies that merge-agent warnings propagate from MergeResolution through
LevelContext into subsequent level prompts, enabling downstream verification
of non-trivial conflict resolutions.
"""

from __future__ import annotations

from typing import Any

import pytest

from ouroboros.orchestrator.events import (
    create_merge_resolution_warnings_event,
)
from ouroboros.orchestrator.level_context import (
    ACContextSummary,
    LevelContext,
    build_context_prompt,
    deserialize_level_contexts,
    serialize_level_contexts,
)
from ouroboros.orchestrator.merge_agent import (
    MergeResolution,
    MergeResolutionOutcome,
    collect_merge_warnings_for_next_level,
)

# ---------------------------------------------------------------------------
# Tests: collect_merge_warnings_for_next_level
# ---------------------------------------------------------------------------


class TestCollectMergeWarningsForNextLevel:
    """Tests for the warning aggregation helper."""

    def test_empty_resolutions_returns_empty(self) -> None:
        result = collect_merge_warnings_for_next_level([])
        assert result == ()

    def test_skipped_resolutions_produce_no_warnings(self) -> None:
        resolutions = [
            MergeResolution(
                ac_index=0,
                ac_branch="ooo/ac_0",
                outcome=MergeResolutionOutcome.SKIPPED,
            ),
        ]
        result = collect_merge_warnings_for_next_level(resolutions)
        assert result == ()

    def test_clean_resolution_no_warnings(self) -> None:
        """Resolved with no warnings produces empty tuple."""
        resolutions = [
            MergeResolution(
                ac_index=0,
                ac_branch="ooo/ac_0",
                outcome=MergeResolutionOutcome.RESOLVED,
                files_resolved=("a.py",),
            ),
        ]
        result = collect_merge_warnings_for_next_level(resolutions)
        assert result == ()

    def test_resolution_warnings_are_prefixed_with_ac_context(self) -> None:
        resolutions = [
            MergeResolution(
                ac_index=2,
                ac_branch="ooo/exec_ac_2",
                outcome=MergeResolutionOutcome.RESOLVED,
                files_resolved=("a.py", "b.py"),
                warnings=(
                    "Merge-agent flagged non-trivial resolution (keyword: incompatible). "
                    "Review recommended.",
                ),
            ),
        ]
        result = collect_merge_warnings_for_next_level(resolutions)
        assert len(result) == 1
        assert "[AC 2, branch ooo/exec_ac_2]" in result[0]
        assert "non-trivial" in result[0]

    def test_partial_resolution_generates_warning(self) -> None:
        resolutions = [
            MergeResolution(
                ac_index=1,
                ac_branch="ooo/ac_1",
                outcome=MergeResolutionOutcome.PARTIAL,
                files_resolved=("a.py",),
                files_remaining=("b.py", "c.py"),
            ),
        ]
        result = collect_merge_warnings_for_next_level(resolutions)
        assert any("partially resolved" in w for w in result)
        assert any("b.py" in w and "c.py" in w for w in result)

    def test_failed_resolution_generates_warning(self) -> None:
        resolutions = [
            MergeResolution(
                ac_index=0,
                ac_branch="ooo/ac_0",
                outcome=MergeResolutionOutcome.FAILED,
                error_message="Agent session crashed",
            ),
        ]
        result = collect_merge_warnings_for_next_level(resolutions)
        assert any("failed" in w.lower() for w in result)
        assert any("Agent session crashed" in w for w in result)

    def test_multiple_resolutions_aggregate_warnings(self) -> None:
        resolutions = [
            MergeResolution(
                ac_index=0,
                ac_branch="ooo/ac_0",
                outcome=MergeResolutionOutcome.RESOLVED,
                files_resolved=("a.py",),
                warnings=("Large conflict scope warning",),
            ),
            MergeResolution(
                ac_index=1,
                ac_branch="ooo/ac_1",
                outcome=MergeResolutionOutcome.PARTIAL,
                files_resolved=("b.py",),
                files_remaining=("c.py",),
                warnings=("Non-trivial keyword warning",),
            ),
        ]
        result = collect_merge_warnings_for_next_level(resolutions)
        # AC 0: 1 warning from resolution
        # AC 1: 1 warning from resolution + 1 for partial
        assert len(result) == 3
        assert any("[AC 0" in w for w in result)
        assert any("[AC 1" in w for w in result)

    def test_failed_with_no_error_message(self) -> None:
        resolutions = [
            MergeResolution(
                ac_index=0,
                ac_branch="ooo/ac_0",
                outcome=MergeResolutionOutcome.FAILED,
            ),
        ]
        result = collect_merge_warnings_for_next_level(resolutions)
        assert any("unknown error" in w for w in result)


# ---------------------------------------------------------------------------
# Tests: LevelContext with merge_warnings
# ---------------------------------------------------------------------------


class TestLevelContextMergeWarnings:
    """Tests for LevelContext merge_warnings field."""

    def test_default_empty(self) -> None:
        ctx = LevelContext(level_number=0)
        assert ctx.merge_warnings == ()

    def test_stores_warnings(self) -> None:
        ctx = LevelContext(
            level_number=1,
            merge_warnings=("Warning 1", "Warning 2"),
        )
        assert len(ctx.merge_warnings) == 2
        assert ctx.merge_warnings[0] == "Warning 1"

    def test_frozen(self) -> None:
        ctx = LevelContext(level_number=0, merge_warnings=("w1",))
        with pytest.raises(AttributeError):
            ctx.merge_warnings = ("w2",)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tests: build_context_prompt with merge_warnings
# ---------------------------------------------------------------------------


class TestBuildContextPromptMergeWarnings:
    """Tests that merge warnings appear in the context prompt."""

    def _make_level_ctx(
        self,
        level: int = 1,
        merge_warnings: tuple[str, ...] = (),
    ) -> LevelContext:
        return LevelContext(
            level_number=level,
            completed_acs=(
                ACContextSummary(
                    ac_index=0,
                    ac_content="Test AC",
                    success=True,
                    files_modified=("test.py",),
                ),
            ),
            merge_warnings=merge_warnings,
        )

    def test_no_merge_warnings_no_section(self) -> None:
        ctx = self._make_level_ctx()
        prompt = build_context_prompt([ctx])
        assert "Merge Resolution Warnings" not in prompt

    def test_merge_warnings_rendered(self) -> None:
        ctx = self._make_level_ctx(
            merge_warnings=(
                "[AC 0, branch ooo/ac_0] Large conflict scope warning",
                "[AC 1, branch ooo/ac_1] Non-trivial resolution detected",
            ),
        )
        prompt = build_context_prompt([ctx])
        assert "Merge Resolution Warnings (Level 1)" in prompt
        assert "Large conflict scope warning" in prompt
        assert "Non-trivial resolution detected" in prompt
        assert "Verify these resolutions" in prompt

    def test_merge_warnings_coexist_with_coordinator_review(self) -> None:
        from ouroboros.orchestrator.coordinator import CoordinatorReview

        review = CoordinatorReview(
            level_number=1,
            review_summary="Coordinator saw issues",
            warnings_for_next_level=("Coordinator warning 1",),
        )
        ctx = LevelContext(
            level_number=1,
            completed_acs=(
                ACContextSummary(
                    ac_index=0, ac_content="AC", success=True,
                ),
            ),
            coordinator_review=review,
            merge_warnings=("Merge warning 1",),
        )
        prompt = build_context_prompt([ctx])
        assert "Coordinator Review" in prompt
        assert "Coordinator warning 1" in prompt
        assert "Merge Resolution Warnings" in prompt
        assert "Merge warning 1" in prompt

    def test_merge_warnings_only_no_coordinator(self) -> None:
        """Context prompt is non-empty when only merge warnings exist."""
        ctx = LevelContext(
            level_number=0,
            completed_acs=(
                ACContextSummary(ac_index=0, ac_content="AC", success=True),
            ),
            merge_warnings=("Some merge warning",),
        )
        prompt = build_context_prompt([ctx])
        assert prompt != ""
        assert "Some merge warning" in prompt

    def test_empty_context_with_only_merge_warnings(self) -> None:
        """Even without successful ACs, merge warnings alone produce output."""
        ctx = LevelContext(
            level_number=0,
            merge_warnings=("Orphan merge warning",),
        )
        prompt = build_context_prompt([ctx])
        assert "Orphan merge warning" in prompt


# ---------------------------------------------------------------------------
# Tests: Serialization round-trip
# ---------------------------------------------------------------------------


class TestSerializationWithMergeWarnings:
    """Tests that merge_warnings survive serialize/deserialize."""

    def test_round_trip(self) -> None:
        original = [
            LevelContext(
                level_number=1,
                completed_acs=(
                    ACContextSummary(
                        ac_index=0, ac_content="AC 1", success=True,
                    ),
                ),
                merge_warnings=(
                    "Warning about non-trivial merge",
                    "Another warning",
                ),
            ),
        ]
        serialized = serialize_level_contexts(original)
        assert list(serialized[0]["merge_warnings"]) == [
            "Warning about non-trivial merge",
            "Another warning",
        ]

        deserialized = deserialize_level_contexts(serialized)
        assert len(deserialized) == 1
        assert deserialized[0].merge_warnings == (
            "Warning about non-trivial merge",
            "Another warning",
        )

    def test_backward_compat_missing_field(self) -> None:
        """Old checkpoints without merge_warnings deserialize cleanly."""
        old_data: list[dict[str, Any]] = [
            {
                "level_number": 0,
                "completed_acs": [],
                # No merge_warnings key
            },
        ]
        result = deserialize_level_contexts(old_data)
        assert result[0].merge_warnings == ()

    def test_empty_warnings_round_trip(self) -> None:
        original = [LevelContext(level_number=0)]
        serialized = serialize_level_contexts(original)
        deserialized = deserialize_level_contexts(serialized)
        assert deserialized[0].merge_warnings == ()


# ---------------------------------------------------------------------------
# Tests: Event creation
# ---------------------------------------------------------------------------


class TestMergeResolutionWarningsEvent:
    """Tests for the merge resolution warnings event factory."""

    def test_creates_event_with_warnings(self) -> None:
        event = create_merge_resolution_warnings_event(
            execution_id="exec_123",
            session_id="sess_456",
            level_number=2,
            warnings=["Warning 1", "Warning 2"],
            resolutions_summary=[
                {"ac_index": 0, "outcome": "resolved", "warnings": ["Warning 1"]},
                {"ac_index": 1, "outcome": "partial", "warnings": ["Warning 2"]},
            ],
        )
        assert event.type == "execution.merge.warnings_flagged"
        assert event.aggregate_type == "execution"
        assert event.aggregate_id == "exec_123"
        assert event.data["session_id"] == "sess_456"
        assert event.data["level_number"] == 2
        assert event.data["warning_count"] == 2
        assert len(event.data["warnings"]) == 2
        assert len(event.data["resolutions_summary"]) == 2

    def test_creates_event_with_empty_warnings(self) -> None:
        event = create_merge_resolution_warnings_event(
            execution_id="exec_1",
            session_id="sess_1",
            level_number=0,
            warnings=[],
            resolutions_summary=[],
        )
        assert event.data["warning_count"] == 0


# ---------------------------------------------------------------------------
# Tests: End-to-end warning flow
# ---------------------------------------------------------------------------


class TestEndToEndWarningFlow:
    """Tests the full flow: MergeResolution → collect → LevelContext → prompt."""

    def test_full_pipeline(self) -> None:
        """Warnings flow from resolutions through context into prompt text."""
        # Step 1: Create resolutions with warnings
        resolutions = [
            MergeResolution(
                ac_index=0,
                ac_branch="ooo/exec_ac_0",
                outcome=MergeResolutionOutcome.RESOLVED,
                files_resolved=("models.py", "views.py", "urls.py", "tests.py"),
                warnings=(
                    "Merge-agent resolved 4 files — large conflict scope increases risk.",
                ),
                agent_summary="Combined both import blocks and route definitions.",
            ),
            MergeResolution(
                ac_index=1,
                ac_branch="ooo/exec_ac_1",
                outcome=MergeResolutionOutcome.RESOLVED,
                files_resolved=("config.py",),
                warnings=(
                    "Merge-agent flagged non-trivial resolution (keyword: chose). "
                    "Review recommended.",
                ),
                agent_summary="Chose AC 1's config format, adapted AC 0's values.",
            ),
        ]

        # Step 2: Collect warnings
        merge_warnings = collect_merge_warnings_for_next_level(resolutions)
        assert len(merge_warnings) == 2

        # Step 3: Attach to LevelContext
        level_ctx = LevelContext(
            level_number=1,
            completed_acs=(
                ACContextSummary(
                    ac_index=0, ac_content="Implement models", success=True,
                    files_modified=("models.py",),
                ),
                ACContextSummary(
                    ac_index=1, ac_content="Add config", success=True,
                    files_modified=("config.py",),
                ),
            ),
            merge_warnings=merge_warnings,
        )

        # Step 4: Build prompt — warnings appear for next level
        prompt = build_context_prompt([level_ctx])
        assert "Merge Resolution Warnings (Level 1)" in prompt
        assert "large conflict scope" in prompt
        assert "non-trivial" in prompt
        assert "[AC 0" in prompt
        assert "[AC 1" in prompt

        # Step 5: Verify serialization round-trip preserves warnings
        serialized = serialize_level_contexts([level_ctx])
        deserialized = deserialize_level_contexts(serialized)
        assert deserialized[0].merge_warnings == merge_warnings
