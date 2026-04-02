"""Tests for the Level Coordinator module.

Tests cover:
- FileConflict and CoordinatorReview data models
- detect_file_conflicts() with various scenarios
- _collect_file_modifications() for atomic and decomposed results
- _build_review_prompt() formatting
- _parse_review_response() JSON parsing and fallback
- build_context_prompt() integration with coordinator_review
"""

from __future__ import annotations

import pytest

from ouroboros.orchestrator.adapter import AgentMessage, RuntimeHandle
from ouroboros.orchestrator.coordinator import (
    ConflictPredictionMetrics,
    CoordinatorReview,
    FileConflict,
    LevelCoordinator,
    _build_review_prompt,
    _collect_file_modifications,
    _parse_review_response,
)
from ouroboros.orchestrator.level_context import (
    ACContextSummary,
    LevelContext,
    build_context_prompt,
)
from ouroboros.orchestrator.parallel_executor import ACExecutionResult

# =============================================================================
# Data Model Tests
# =============================================================================


class TestFileConflict:
    """Tests for FileConflict dataclass."""

    def test_basic_creation(self):
        conflict = FileConflict(
            file_path="src/app.py",
            ac_indices=(0, 2),
        )
        assert conflict.file_path == "src/app.py"
        assert conflict.ac_indices == (0, 2)
        assert conflict.resolved is False
        assert conflict.resolution_description == ""

    def test_resolved_conflict(self):
        conflict = FileConflict(
            file_path="src/app.py",
            ac_indices=(0, 1),
            resolved=True,
            resolution_description="Merged imports from both ACs",
        )
        assert conflict.resolved is True
        assert conflict.resolution_description == "Merged imports from both ACs"

    def test_frozen(self):
        conflict = FileConflict(file_path="a.py", ac_indices=(0,))
        with pytest.raises(AttributeError):
            conflict.file_path = "b.py"


class TestCoordinatorReview:
    """Tests for CoordinatorReview dataclass."""

    def test_basic_creation(self):
        review = CoordinatorReview(level_number=1)
        assert review.level_number == 1
        assert review.conflicts_detected == ()
        assert review.review_summary == ""
        assert review.fixes_applied == ()
        assert review.warnings_for_next_level == ()
        assert review.duration_seconds == 0.0
        assert review.session_id is None
        assert review.session_scope_id is None
        assert review.session_state_path is None
        assert review.scope == "level"
        assert review.session_role == "coordinator"
        assert review.stage_index == 0
        assert review.artifact_scope == "level"
        assert review.artifact_owner == "coordinator"
        assert review.artifact_type == "coordinator_review"
        assert review.artifact_owner_id == "level_1_coordinator_reconciliation"
        assert (
            review.artifact_state_path
            == "execution.levels.level_1.coordinator_reconciliation_session"
        )

    def test_full_review(self):
        conflict = FileConflict(
            file_path="src/routes.py",
            ac_indices=(0, 1),
            resolved=True,
        )
        review = CoordinatorReview(
            level_number=2,
            conflicts_detected=(conflict,),
            review_summary="Resolved import conflict in routes.py",
            fixes_applied=("Merged duplicate import statements",),
            warnings_for_next_level=("Ensure routes are registered in main.py",),
            duration_seconds=5.3,
            session_id="sess_abc",
            session_scope_id="exec_scope_level_2_coordinator_reconciliation",
            session_state_path=(
                "execution.workflows.exec_scope.levels.level_2.coordinator_reconciliation_session"
            ),
        )
        assert len(review.conflicts_detected) == 1
        assert review.conflicts_detected[0].resolved is True
        assert len(review.fixes_applied) == 1
        assert len(review.warnings_for_next_level) == 1
        assert review.session_scope_id == "exec_scope_level_2_coordinator_reconciliation"
        assert (
            review.session_state_path == "execution.workflows.exec_scope.levels.level_2."
            "coordinator_reconciliation_session"
        )
        assert review.artifact_owner_id == "exec_scope_level_2_coordinator_reconciliation"
        assert (
            review.artifact_state_path == "execution.workflows.exec_scope.levels.level_2."
            "coordinator_reconciliation_session"
        )

    def test_artifact_payload_is_explicitly_level_scoped(self):
        review = CoordinatorReview(
            level_number=3,
            session_scope_id="level_3_coordinator_reconciliation",
            session_state_path="execution.levels.level_3.coordinator_reconciliation_session",
            final_output='{"review_summary":"resolved"}',
        )

        assert review.to_artifact_payload() == {
            "scope": "level",
            "session_role": "coordinator",
            "stage_index": 2,
            "level_number": 3,
            "session_scope_id": "level_3_coordinator_reconciliation",
            "session_state_path": "execution.levels.level_3.coordinator_reconciliation_session",
            "artifact_scope": "level",
            "artifact_owner": "coordinator",
            "artifact_owner_id": "level_3_coordinator_reconciliation",
            "artifact": '{"review_summary":"resolved"}',
            "artifact_type": "coordinator_review",
        }

    def test_frozen(self):
        review = CoordinatorReview(level_number=1)
        with pytest.raises(AttributeError):
            review.level_number = 2


# =============================================================================
# detect_file_conflicts Tests
# =============================================================================


def _make_result(
    ac_index: int,
    tool_calls: list[tuple[str, str]] | None = None,
    sub_results: list[ACExecutionResult] | None = None,
) -> ACExecutionResult:
    """Helper to create ACExecutionResult with specific tool calls.

    Args:
        ac_index: AC index.
        tool_calls: List of (tool_name, file_path) tuples.
        sub_results: Optional sub-results for decomposed ACs.
    """
    messages = []
    for tool_name, file_path in tool_calls or []:
        messages.append(
            AgentMessage(
                type="assistant",
                content=f"Using {tool_name}",
                tool_name=tool_name,
                data={"tool_input": {"file_path": file_path}},
            )
        )
    return ACExecutionResult(
        ac_index=ac_index,
        ac_content=f"AC {ac_index + 1} content",
        success=True,
        messages=tuple(messages),
        sub_results=tuple(sub_results or []),
    )


class _StubCoordinatorRuntime:
    """Minimal runtime stub for coordinator review tests."""

    def __init__(self, messages: tuple[AgentMessage, ...]) -> None:
        self._messages = messages
        self.calls: list[dict[str, object]] = []
        self._runtime_handle_backend = "opencode"
        self._cwd = "/tmp/project"
        self._permission_mode = "acceptEdits"

    @property
    def runtime_backend(self) -> str:
        return self._runtime_handle_backend

    @property
    def working_directory(self) -> str | None:
        return self._cwd

    @property
    def permission_mode(self) -> str | None:
        return self._permission_mode

    async def execute_task(
        self,
        prompt: str,
        tools: list[str] | None = None,
        system_prompt: str | None = None,
        resume_handle: RuntimeHandle | None = None,
        resume_session_id: str | None = None,
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "tools": tools,
                "system_prompt": system_prompt,
                "resume_handle": resume_handle,
                "resume_session_id": resume_session_id,
            }
        )
        for message in self._messages:
            yield message


class TestDetectFileConflicts:
    """Tests for LevelCoordinator.detect_file_conflicts()."""

    def test_no_results(self):
        conflicts = LevelCoordinator.detect_file_conflicts([])
        assert conflicts == []

    def test_no_conflicts_different_files(self):
        results = [
            _make_result(0, [("Write", "src/a.py")]),
            _make_result(1, [("Edit", "src/b.py")]),
        ]
        conflicts = LevelCoordinator.detect_file_conflicts(results)
        assert conflicts == []

    def test_single_conflict(self):
        results = [
            _make_result(0, [("Write", "src/app.py")]),
            _make_result(1, [("Edit", "src/app.py")]),
        ]
        conflicts = LevelCoordinator.detect_file_conflicts(results)
        assert len(conflicts) == 1
        assert conflicts[0].file_path == "src/app.py"
        assert conflicts[0].ac_indices == (0, 1)
        assert conflicts[0].resolved is False

    def test_multiple_conflicts(self):
        results = [
            _make_result(0, [("Write", "src/a.py"), ("Edit", "src/b.py")]),
            _make_result(1, [("Edit", "src/a.py")]),
            _make_result(2, [("Write", "src/b.py"), ("Write", "src/c.py")]),
        ]
        conflicts = LevelCoordinator.detect_file_conflicts(results)
        assert len(conflicts) == 2
        # Sorted by file path
        assert conflicts[0].file_path == "src/a.py"
        assert conflicts[0].ac_indices == (0, 1)
        assert conflicts[1].file_path == "src/b.py"
        assert conflicts[1].ac_indices == (0, 2)

    def test_three_way_conflict(self):
        results = [
            _make_result(0, [("Write", "src/shared.py")]),
            _make_result(1, [("Edit", "src/shared.py")]),
            _make_result(2, [("Edit", "src/shared.py")]),
        ]
        conflicts = LevelCoordinator.detect_file_conflicts(results)
        assert len(conflicts) == 1
        assert conflicts[0].ac_indices == (0, 1, 2)

    def test_ignores_read_and_other_tools(self):
        results = [
            _make_result(
                0,
                [("Write", "src/app.py")],
            ),
            ACExecutionResult(
                ac_index=1,
                ac_content="AC 2",
                success=True,
                messages=(
                    AgentMessage(
                        type="assistant",
                        content="Reading",
                        tool_name="Read",
                        data={"tool_input": {"file_path": "src/app.py"}},
                    ),
                    AgentMessage(
                        type="assistant",
                        content="Grepping",
                        tool_name="Grep",
                        data={"tool_input": {"pattern": "import"}},
                    ),
                ),
            ),
        ]
        conflicts = LevelCoordinator.detect_file_conflicts(results)
        assert conflicts == []

    def test_same_ac_multiple_edits_no_conflict(self):
        """Same AC editing same file multiple times is NOT a conflict."""
        results = [
            _make_result(0, [("Write", "src/app.py"), ("Edit", "src/app.py")]),
        ]
        conflicts = LevelCoordinator.detect_file_conflicts(results)
        assert conflicts == []

    def test_decomposed_sub_acs_inherit_parent_index(self):
        """Sub-AC modifications are attributed to the parent AC index."""
        sub_result = ACExecutionResult(
            ac_index=100,  # Sub-AC index (parent * 100 + sub)
            ac_content="Sub-AC 1",
            success=True,
            messages=(
                AgentMessage(
                    type="assistant",
                    content="Writing",
                    tool_name="Write",
                    data={"tool_input": {"file_path": "src/shared.py"}},
                ),
            ),
        )
        results = [
            _make_result(0, sub_results=[sub_result]),
            _make_result(1, [("Edit", "src/shared.py")]),
        ]
        conflicts = LevelCoordinator.detect_file_conflicts(results)
        assert len(conflicts) == 1
        assert conflicts[0].ac_indices == (0, 1)

    def test_no_file_path_in_tool_input(self):
        """Messages without file_path in tool_input are safely ignored."""
        results = [
            ACExecutionResult(
                ac_index=0,
                ac_content="AC 1",
                success=True,
                messages=(
                    AgentMessage(
                        type="assistant",
                        content="Writing",
                        tool_name="Write",
                        data={"tool_input": {}},  # No file_path
                    ),
                ),
            ),
        ]
        conflicts = LevelCoordinator.detect_file_conflicts(results)
        assert conflicts == []


# =============================================================================
# _collect_file_modifications Tests
# =============================================================================


class TestCollectFileModifications:
    """Tests for _collect_file_modifications helper."""

    def test_empty_messages(self):
        result = _make_result(0)
        acc: dict[str, set[int]] = {}
        _collect_file_modifications(result, acc)
        assert acc == {}

    def test_write_and_edit(self):
        result = _make_result(0, [("Write", "a.py"), ("Edit", "b.py")])
        acc: dict[str, set[int]] = {}
        _collect_file_modifications(result, acc)
        assert acc == {"a.py": {0}, "b.py": {0}}

    def test_nested_sub_results(self):
        sub = ACExecutionResult(
            ac_index=100,
            ac_content="sub",
            success=True,
            messages=(
                AgentMessage(
                    type="assistant",
                    content="w",
                    tool_name="Edit",
                    data={"tool_input": {"file_path": "deep.py"}},
                ),
            ),
        )
        parent = _make_result(0, [("Write", "top.py")], sub_results=[sub])
        acc: dict[str, set[int]] = {}
        _collect_file_modifications(parent, acc)
        assert acc == {"top.py": {0}, "deep.py": {0}}


# =============================================================================
# _build_review_prompt Tests
# =============================================================================


class TestBuildReviewPrompt:
    """Tests for _build_review_prompt."""

    def test_basic_prompt(self):
        conflicts = [
            FileConflict(file_path="src/app.py", ac_indices=(0, 1)),
        ]
        level_ctx = LevelContext(
            level_number=1,
            completed_acs=(
                ACContextSummary(ac_index=0, ac_content="Create model", success=True),
                ACContextSummary(ac_index=1, ac_content="Create routes", success=True),
            ),
        )
        prompt = _build_review_prompt(conflicts, level_ctx, 1)

        assert "Level 1" in prompt
        assert "src/app.py" in prompt
        assert "AC 1" in prompt
        assert "AC 2" in prompt
        assert "Read tool" in prompt
        assert "git diff" in prompt

    def test_multiple_conflicts(self):
        conflicts = [
            FileConflict(file_path="a.py", ac_indices=(0, 2)),
            FileConflict(file_path="b.py", ac_indices=(1, 2)),
        ]
        level_ctx = LevelContext(level_number=2, completed_acs=())
        prompt = _build_review_prompt(conflicts, level_ctx, 2)
        assert "a.py" in prompt
        assert "b.py" in prompt


# =============================================================================
# _parse_review_response Tests
# =============================================================================


class TestParseReviewResponse:
    """Tests for _parse_review_response."""

    def test_valid_json_response(self):
        response = """I've reviewed the conflicts.

```json
{
  "review_summary": "Merged duplicate imports",
  "fixes_applied": ["Combined import statements in app.py"],
  "warnings_for_next_level": ["Check route registration"],
  "conflicts_resolved": ["src/app.py"]
}
```
"""
        conflicts = [FileConflict(file_path="src/app.py", ac_indices=(0, 1))]
        review = _parse_review_response(response, conflicts, 1, 3.5, "sess_1")

        assert review.level_number == 1
        assert review.review_summary == "Merged duplicate imports"
        assert review.fixes_applied == ("Combined import statements in app.py",)
        assert review.warnings_for_next_level == ("Check route registration",)
        assert review.duration_seconds == 3.5
        assert review.session_id == "sess_1"
        assert review.conflicts_detected[0].resolved is True

    def test_carries_session_scope_metadata(self):
        response = '{"review_summary": "Scoped review", "conflicts_resolved": []}'
        review = _parse_review_response(
            response,
            [],
            2,
            1.25,
            "sess_2",
            session_scope_id="exec_scope_level_2_coordinator_reconciliation",
            session_state_path=(
                "execution.workflows.exec_scope.levels.level_2.coordinator_reconciliation_session"
            ),
        )

        assert review.session_scope_id == "exec_scope_level_2_coordinator_reconciliation"
        assert (
            review.session_state_path == "execution.workflows.exec_scope.levels.level_2."
            "coordinator_reconciliation_session"
        )

    def test_bare_json_response(self):
        response = '{"review_summary": "All good", "fixes_applied": [], "warnings_for_next_level": [], "conflicts_resolved": []}'
        review = _parse_review_response(response, [], 2, 1.0, None)
        assert review.review_summary == "All good"

    def test_invalid_json_fallback(self):
        response = "I couldn't parse anything, but here's my review."
        review = _parse_review_response(response, [], 1, 2.0, None)
        assert review.review_summary == response

    def test_empty_response_fallback(self):
        review = _parse_review_response("", [], 1, 0.5, None)
        assert review.review_summary == "No review output"

    def test_unresolved_conflicts_stay_unresolved(self):
        response = '```json\n{"review_summary": "x", "fixes_applied": [], "warnings_for_next_level": [], "conflicts_resolved": ["a.py"]}\n```'
        conflicts = [
            FileConflict(file_path="a.py", ac_indices=(0, 1)),
            FileConflict(file_path="b.py", ac_indices=(0, 2)),
        ]
        review = _parse_review_response(response, conflicts, 1, 1.0, None)
        assert review.conflicts_detected[0].resolved is True
        assert review.conflicts_detected[1].resolved is False

    def test_partial_json_missing_fields(self):
        response = '```json\n{"review_summary": "partial"}\n```'
        review = _parse_review_response(response, [], 1, 1.0, None)
        assert review.review_summary == "partial"
        assert review.fixes_applied == ()
        assert review.warnings_for_next_level == ()


class TestRunReview:
    """Tests for LevelCoordinator.run_review()."""

    @pytest.mark.asyncio
    async def test_run_review_uses_fresh_level_scoped_runtime_handle(self):
        runtime = _StubCoordinatorRuntime(
            (
                AgentMessage(
                    type="assistant",
                    content="Reviewing conflicts",
                    resume_handle=RuntimeHandle(
                        backend="opencode",
                        kind="level_coordinator",
                        native_session_id="coord-level-1",
                        cwd="/tmp/project",
                        approval_mode="acceptEdits",
                        metadata={
                            "scope": "level",
                            "level_number": 1,
                            "session_role": "coordinator",
                        },
                    ),
                ),
                AgentMessage(
                    type="result",
                    content='{"review_summary":"Resolved","fixes_applied":[],"warnings_for_next_level":[],"conflicts_resolved":[]}',
                    data={"subtype": "success"},
                    resume_handle=RuntimeHandle(
                        backend="opencode",
                        kind="level_coordinator",
                        native_session_id="coord-level-1",
                        cwd="/tmp/project",
                        approval_mode="acceptEdits",
                        metadata={
                            "scope": "level",
                            "level_number": 1,
                            "session_role": "coordinator",
                        },
                    ),
                ),
            )
        )
        coordinator = LevelCoordinator(runtime)
        level_ctx = LevelContext(level_number=1, completed_acs=())

        review = await coordinator.run_review(
            execution_id="exec_level_scope",
            conflicts=[FileConflict(file_path="src/app.py", ac_indices=(0, 1))],
            level_context=level_ctx,
            level_number=1,
        )

        assert review.review_summary == "Resolved"
        assert review.session_id == "coord-level-1"
        assert review.session_scope_id == "exec_level_scope_level_1_coordinator_reconciliation"
        assert (
            review.session_state_path == "execution.workflows.exec_level_scope.levels.level_1."
            "coordinator_reconciliation_session"
        )
        assert len(runtime.calls) == 1
        resume_handle = runtime.calls[0]["resume_handle"]
        assert isinstance(resume_handle, RuntimeHandle)
        assert resume_handle.native_session_id is None
        assert resume_handle.backend == "opencode"
        assert resume_handle.kind == "level_coordinator"
        assert resume_handle.cwd == "/tmp/project"
        assert resume_handle.approval_mode == "acceptEdits"
        assert resume_handle.metadata["scope"] == "level"
        assert resume_handle.metadata["execution_id"] == "exec_level_scope"
        assert resume_handle.metadata["level_number"] == 1
        assert resume_handle.metadata["session_role"] == "coordinator"
        assert (
            resume_handle.metadata["session_scope_id"]
            == "exec_level_scope_level_1_coordinator_reconciliation"
        )
        assert (
            resume_handle.metadata["session_state_path"]
            == "execution.workflows.exec_level_scope.levels.level_1."
            "coordinator_reconciliation_session"
        )


# =============================================================================
# build_context_prompt Integration with CoordinatorReview
# =============================================================================


class TestBuildContextPromptWithReview:
    """Tests that build_context_prompt() includes coordinator review."""

    def test_no_review(self):
        contexts = [
            LevelContext(
                level_number=1,
                completed_acs=(ACContextSummary(ac_index=0, ac_content="AC 1", success=True),),
            ),
        ]
        prompt = build_context_prompt(contexts)
        assert "Previous Work Context" in prompt
        assert "Coordinator Review" not in prompt

    def test_with_review(self):
        review = CoordinatorReview(
            level_number=1,
            review_summary="Fixed merge conflict in app.py",
            fixes_applied=("Merged imports",),
            warnings_for_next_level=("Register new routes in main.py",),
        )
        contexts = [
            LevelContext(
                level_number=1,
                completed_acs=(ACContextSummary(ac_index=0, ac_content="AC 1", success=True),),
                coordinator_review=review,
            ),
        ]
        prompt = build_context_prompt(contexts)
        assert "Coordinator Review (Level 1)" in prompt
        assert "Fixed merge conflict in app.py" in prompt
        assert "Merged imports" in prompt
        assert "WARNING: Register new routes in main.py" in prompt

    def test_review_with_empty_warnings(self):
        review = CoordinatorReview(
            level_number=1,
            review_summary="No issues found",
        )
        contexts = [
            LevelContext(
                level_number=1,
                completed_acs=(ACContextSummary(ac_index=0, ac_content="AC 1", success=True),),
                coordinator_review=review,
            ),
        ]
        prompt = build_context_prompt(contexts)
        assert "No issues found" in prompt
        assert "WARNING" not in prompt

    def test_multiple_levels_with_mixed_reviews(self):
        """Only levels with reviews include review sections."""
        review = CoordinatorReview(
            level_number=2,
            review_summary="Conflict resolved",
            warnings_for_next_level=("Watch out for X",),
        )
        contexts = [
            LevelContext(
                level_number=1,
                completed_acs=(ACContextSummary(ac_index=0, ac_content="AC 1", success=True),),
                # No coordinator_review
            ),
            LevelContext(
                level_number=2,
                completed_acs=(ACContextSummary(ac_index=1, ac_content="AC 2", success=True),),
                coordinator_review=review,
            ),
        ]
        prompt = build_context_prompt(contexts)
        assert "Coordinator Review (Level 2)" in prompt
        assert "Coordinator Review (Level 1)" not in prompt
        assert "Watch out for X" in prompt


# =============================================================================
# ConflictPredictionMetrics Tests
# =============================================================================


class TestConflictPredictionMetrics:
    """Tests for ConflictPredictionMetrics dataclass."""

    def test_default_empty_metrics(self):
        metrics = ConflictPredictionMetrics()
        assert metrics.level_number == 0
        assert metrics.predicted_overlap_files == frozenset()
        assert metrics.actual_conflict_files == frozenset()
        assert metrics.covered_conflicts == frozenset()
        assert metrics.missed_conflicts == frozenset()
        assert metrics.false_positive_files == frozenset()
        assert metrics.prediction_was_active is False
        assert metrics.isolation_was_applied is False

    def test_prediction_covered_all_when_no_misses(self):
        metrics = ConflictPredictionMetrics(
            level_number=1,
            predicted_overlap_files=frozenset(["a.py", "b.py"]),
            actual_conflict_files=frozenset(["a.py"]),
            covered_conflicts=frozenset(["a.py"]),
            missed_conflicts=frozenset(),
            false_positive_files=frozenset(["b.py"]),
            prediction_was_active=True,
        )
        assert metrics.prediction_covered_all is True
        assert metrics.safety_net_fired is False

    def test_safety_net_fired_when_misses_exist(self):
        metrics = ConflictPredictionMetrics(
            level_number=2,
            predicted_overlap_files=frozenset(["a.py"]),
            actual_conflict_files=frozenset(["a.py", "c.py"]),
            covered_conflicts=frozenset(["a.py"]),
            missed_conflicts=frozenset(["c.py"]),
            prediction_was_active=True,
        )
        assert metrics.prediction_covered_all is False
        assert metrics.safety_net_fired is True

    def test_to_metadata_serialization(self):
        metrics = ConflictPredictionMetrics(
            level_number=1,
            predicted_overlap_files=frozenset(["a.py", "b.py"]),
            actual_conflict_files=frozenset(["a.py"]),
            covered_conflicts=frozenset(["a.py"]),
            missed_conflicts=frozenset(),
            false_positive_files=frozenset(["b.py"]),
            prediction_was_active=True,
            isolation_was_applied=True,
        )
        meta = metrics.to_metadata()
        assert meta["level_number"] == 1
        assert meta["predicted_overlap_count"] == 2
        assert meta["actual_conflict_count"] == 1
        assert meta["covered_count"] == 1
        assert meta["missed_count"] == 0
        assert meta["false_positive_count"] == 1
        assert meta["prediction_was_active"] is True
        assert meta["isolation_was_applied"] is True
        assert meta["prediction_covered_all"] is True
        assert meta["safety_net_fired"] is False
        assert meta["missed_files"] == []

    def test_to_metadata_with_missed_files(self):
        metrics = ConflictPredictionMetrics(
            level_number=3,
            missed_conflicts=frozenset(["x.py", "a.py"]),
        )
        meta = metrics.to_metadata()
        assert meta["missed_files"] == ["a.py", "x.py"]  # sorted


# =============================================================================
# build_prediction_metrics Tests
# =============================================================================


class TestBuildPredictionMetrics:
    """Tests for LevelCoordinator.build_prediction_metrics()."""

    def test_no_prediction_no_conflicts(self):
        metrics = LevelCoordinator.build_prediction_metrics(
            level_number=1,
            prediction=None,
            conflicts=[],
        )
        assert metrics.prediction_was_active is False
        assert metrics.prediction_covered_all is True  # no conflicts to miss
        assert metrics.safety_net_fired is False

    def test_no_prediction_with_conflicts(self):
        """When prediction is None, all conflicts are 'missed'."""
        conflicts = [
            FileConflict(file_path="a.py", ac_indices=(0, 1)),
            FileConflict(file_path="b.py", ac_indices=(0, 2)),
        ]
        metrics = LevelCoordinator.build_prediction_metrics(
            level_number=1,
            prediction=None,
            conflicts=conflicts,
        )
        assert metrics.prediction_was_active is False
        assert metrics.missed_conflicts == frozenset(["a.py", "b.py"])
        assert metrics.safety_net_fired is True

    def test_prediction_covers_all_conflicts(self):
        """Prediction covered every actual conflict — no safety net needed."""
        from ouroboros.orchestrator.file_overlap_predictor import (
            ACFilePrediction,
            FileOverlapPrediction,
            OverlapGroup,
        )

        prediction = FileOverlapPrediction(
            ac_predictions=(
                ACFilePrediction(ac_index=0, predicted_paths=frozenset(["a.py", "b.py"])),
                ACFilePrediction(ac_index=1, predicted_paths=frozenset(["a.py", "c.py"])),
            ),
            overlap_groups=(
                OverlapGroup(ac_indices=(0, 1), shared_paths=frozenset(["a.py"])),
            ),
            isolated_ac_indices=frozenset({0, 1}),
            shared_ac_indices=frozenset(),
        )
        conflicts = [FileConflict(file_path="a.py", ac_indices=(0, 1))]

        metrics = LevelCoordinator.build_prediction_metrics(
            level_number=1,
            prediction=prediction,
            conflicts=conflicts,
            isolation_was_applied=True,
        )
        assert metrics.prediction_was_active is True
        assert metrics.covered_conflicts == frozenset(["a.py"])
        assert metrics.missed_conflicts == frozenset()
        assert metrics.prediction_covered_all is True
        assert metrics.safety_net_fired is False
        assert metrics.isolation_was_applied is True

    def test_prediction_misses_some_conflicts(self):
        """Prediction missed a conflict — safety net fires."""
        from ouroboros.orchestrator.file_overlap_predictor import (
            ACFilePrediction,
            FileOverlapPrediction,
            OverlapGroup,
        )

        prediction = FileOverlapPrediction(
            ac_predictions=(
                ACFilePrediction(ac_index=0, predicted_paths=frozenset(["a.py"])),
                ACFilePrediction(ac_index=1, predicted_paths=frozenset(["a.py"])),
            ),
            overlap_groups=(
                OverlapGroup(ac_indices=(0, 1), shared_paths=frozenset(["a.py"])),
            ),
            isolated_ac_indices=frozenset({0, 1}),
            shared_ac_indices=frozenset(),
        )
        # a.py was predicted, but b.py was not
        conflicts = [
            FileConflict(file_path="a.py", ac_indices=(0, 1)),
            FileConflict(file_path="b.py", ac_indices=(0, 1)),
        ]

        metrics = LevelCoordinator.build_prediction_metrics(
            level_number=2,
            prediction=prediction,
            conflicts=conflicts,
        )
        assert metrics.prediction_was_active is True
        assert metrics.covered_conflicts == frozenset(["a.py"])
        assert metrics.missed_conflicts == frozenset(["b.py"])
        assert metrics.safety_net_fired is True

    def test_false_positives_tracked(self):
        """Predicted overlaps that didn't become actual conflicts."""
        from ouroboros.orchestrator.file_overlap_predictor import (
            ACFilePrediction,
            FileOverlapPrediction,
            OverlapGroup,
        )

        prediction = FileOverlapPrediction(
            ac_predictions=(
                ACFilePrediction(ac_index=0, predicted_paths=frozenset(["a.py", "b.py", "c.py"])),
                ACFilePrediction(ac_index=1, predicted_paths=frozenset(["a.py", "b.py", "c.py"])),
            ),
            overlap_groups=(
                OverlapGroup(
                    ac_indices=(0, 1),
                    shared_paths=frozenset(["a.py", "b.py", "c.py"]),
                ),
            ),
            isolated_ac_indices=frozenset({0, 1}),
            shared_ac_indices=frozenset(),
        )
        # Only a.py had an actual conflict
        conflicts = [FileConflict(file_path="a.py", ac_indices=(0, 1))]

        metrics = LevelCoordinator.build_prediction_metrics(
            level_number=1,
            prediction=prediction,
            conflicts=conflicts,
        )
        assert metrics.false_positive_files == frozenset(["b.py", "c.py"])
        assert metrics.prediction_covered_all is True

    def test_prediction_no_overlaps_no_conflicts(self):
        """Clean run: prediction found no overlaps, no conflicts occurred."""
        from ouroboros.orchestrator.file_overlap_predictor import (
            ACFilePrediction,
            FileOverlapPrediction,
        )

        prediction = FileOverlapPrediction(
            ac_predictions=(
                ACFilePrediction(ac_index=0, predicted_paths=frozenset(["a.py"])),
                ACFilePrediction(ac_index=1, predicted_paths=frozenset(["b.py"])),
            ),
            overlap_groups=(),
            isolated_ac_indices=frozenset(),
            shared_ac_indices=frozenset({0, 1}),
        )

        metrics = LevelCoordinator.build_prediction_metrics(
            level_number=1,
            prediction=prediction,
            conflicts=[],
        )
        assert metrics.prediction_was_active is True
        assert metrics.prediction_covered_all is True
        assert metrics.safety_net_fired is False
        assert metrics.false_positive_files == frozenset()


# =============================================================================
# Safety Net Fallback Integration Tests
# =============================================================================


class TestCoordinatorSafetyNetFallback:
    """Verify coordinator post-hoc detection remains as fallback for rare false negatives.

    The proactive file overlap prediction engine runs before execution and isolates
    ACs into worktrees when overlap is predicted. However, prediction can have rare
    false negatives. The coordinator's detect_file_conflicts() always runs after
    execution — it is the safety net that catches anything prediction missed.

    These tests validate the full fallback chain:
    1. Prediction runs but misses some file overlaps
    2. Coordinator post-hoc detection catches the missed conflicts
    3. Metrics correctly reflect safety_net_fired=True
    4. Review is triggered for the missed conflicts
    """

    def test_safety_net_catches_unpredicted_conflict(self):
        """Coordinator detects a conflict that prediction did not flag."""
        # AC 0 and AC 1 both edited shared.py, but prediction only flagged config.py
        results = [
            _make_result(0, [("Edit", "src/shared.py"), ("Edit", "src/config.py")]),
            _make_result(1, [("Edit", "src/shared.py"), ("Write", "src/other.py")]),
        ]

        # Post-hoc detection catches it
        conflicts = LevelCoordinator.detect_file_conflicts(results)
        assert len(conflicts) == 1
        assert conflicts[0].file_path == "src/shared.py"
        assert conflicts[0].ac_indices == (0, 1)

    def test_safety_net_fires_when_prediction_misses_file(self):
        """Prediction covered some overlaps but missed one — safety net fires."""
        from ouroboros.orchestrator.file_overlap_predictor import (
            ACFilePrediction,
            FileOverlapPrediction,
            OverlapGroup,
        )

        # Prediction flagged config.py as overlapping, but missed shared.py
        prediction = FileOverlapPrediction(
            ac_predictions=(
                ACFilePrediction(
                    ac_index=0,
                    predicted_paths=frozenset(["src/config.py", "src/a.py"]),
                ),
                ACFilePrediction(
                    ac_index=1,
                    predicted_paths=frozenset(["src/config.py", "src/b.py"]),
                ),
            ),
            overlap_groups=(
                OverlapGroup(
                    ac_indices=(0, 1),
                    shared_paths=frozenset(["src/config.py"]),
                ),
            ),
            isolated_ac_indices=frozenset({0, 1}),
            shared_ac_indices=frozenset(),
        )

        # Actual conflicts: config.py (predicted) and shared.py (missed)
        actual_conflicts = [
            FileConflict(file_path="src/config.py", ac_indices=(0, 1)),
            FileConflict(file_path="src/shared.py", ac_indices=(0, 1)),
        ]

        metrics = LevelCoordinator.build_prediction_metrics(
            level_number=1,
            prediction=prediction,
            conflicts=actual_conflicts,
            isolation_was_applied=True,
        )

        assert metrics.safety_net_fired is True
        assert metrics.missed_conflicts == frozenset(["src/shared.py"])
        assert metrics.covered_conflicts == frozenset(["src/config.py"])
        assert metrics.prediction_was_active is True
        assert metrics.isolation_was_applied is True

    def test_safety_net_does_not_fire_when_prediction_covers_all(self):
        """When prediction covers all actual conflicts, safety net stays silent."""
        from ouroboros.orchestrator.file_overlap_predictor import (
            ACFilePrediction,
            FileOverlapPrediction,
            OverlapGroup,
        )

        prediction = FileOverlapPrediction(
            ac_predictions=(
                ACFilePrediction(
                    ac_index=0,
                    predicted_paths=frozenset(["src/shared.py", "src/config.py"]),
                ),
                ACFilePrediction(
                    ac_index=1,
                    predicted_paths=frozenset(["src/shared.py", "src/config.py"]),
                ),
            ),
            overlap_groups=(
                OverlapGroup(
                    ac_indices=(0, 1),
                    shared_paths=frozenset(["src/shared.py", "src/config.py"]),
                ),
            ),
            isolated_ac_indices=frozenset({0, 1}),
            shared_ac_indices=frozenset(),
        )

        actual_conflicts = [
            FileConflict(file_path="src/shared.py", ac_indices=(0, 1)),
        ]

        metrics = LevelCoordinator.build_prediction_metrics(
            level_number=1,
            prediction=prediction,
            conflicts=actual_conflicts,
            isolation_was_applied=True,
        )

        assert metrics.safety_net_fired is False
        assert metrics.prediction_covered_all is True
        assert metrics.covered_conflicts == frozenset(["src/shared.py"])
        assert metrics.false_positive_files == frozenset(["src/config.py"])

    def test_detect_file_conflicts_always_runs_regardless_of_prediction(self):
        """detect_file_conflicts is a static method with no prediction awareness.

        This verifies the architectural property: post-hoc detection is completely
        independent of pre-execution prediction. It always scans actual tool call
        messages, ensuring it catches conflicts regardless of prediction state.
        """
        # Scenario: even when ACs ran in worktrees, if both touched the same file
        # the coordinator detects it from the merged result messages
        results = [
            _make_result(0, [("Write", "src/api.py"), ("Edit", "src/models.py")]),
            _make_result(1, [("Edit", "src/models.py")]),
            _make_result(2, [("Write", "src/views.py")]),
        ]

        conflicts = LevelCoordinator.detect_file_conflicts(results)

        # Only models.py is a conflict (touched by AC 0 and AC 1)
        assert len(conflicts) == 1
        assert conflicts[0].file_path == "src/models.py"
        assert conflicts[0].ac_indices == (0, 1)

    def test_safety_net_metrics_serialization_includes_missed_files(self):
        """Missed files are included in serialized metadata for observability."""
        metrics = ConflictPredictionMetrics(
            level_number=2,
            predicted_overlap_files=frozenset(["a.py"]),
            actual_conflict_files=frozenset(["a.py", "missed.py", "also_missed.py"]),
            covered_conflicts=frozenset(["a.py"]),
            missed_conflicts=frozenset(["missed.py", "also_missed.py"]),
            false_positive_files=frozenset(),
            prediction_was_active=True,
            isolation_was_applied=True,
        )

        meta = metrics.to_metadata()
        assert meta["safety_net_fired"] is True
        assert meta["missed_count"] == 2
        assert meta["missed_files"] == ["also_missed.py", "missed.py"]  # sorted
        assert meta["prediction_covered_all"] is False

    @pytest.mark.asyncio
    async def test_review_triggered_for_missed_conflicts(self):
        """When safety net detects missed conflicts, run_review resolves them."""
        review_response = (
            '{"review_summary": "Resolved missed conflict in shared.py",'
            ' "fixes_applied": ["Merged overlapping edits"],'
            ' "warnings_for_next_level": ["Verify shared.py integration"],'
            ' "conflicts_resolved": ["src/shared.py"]}'
        )
        runtime = _StubCoordinatorRuntime(
            (
                AgentMessage(
                    type="assistant",
                    content="Inspecting conflict",
                    resume_handle=RuntimeHandle(
                        backend="opencode",
                        kind="level_coordinator",
                        native_session_id="safety-net-review-1",
                        cwd="/tmp/project",
                        approval_mode="acceptEdits",
                        metadata={},
                    ),
                ),
                AgentMessage(
                    type="result",
                    content=review_response,
                    data={"subtype": "success"},
                    resume_handle=RuntimeHandle(
                        backend="opencode",
                        kind="level_coordinator",
                        native_session_id="safety-net-review-1",
                        cwd="/tmp/project",
                        approval_mode="acceptEdits",
                        metadata={},
                    ),
                ),
            )
        )
        coordinator = LevelCoordinator(runtime)
        level_ctx = LevelContext(
            level_number=1,
            completed_acs=(
                ACContextSummary(ac_index=0, ac_content="AC 1", success=True),
                ACContextSummary(ac_index=1, ac_content="AC 2", success=True),
            ),
        )

        # Conflicts detected by safety net (prediction missed src/shared.py)
        missed_conflicts = [
            FileConflict(file_path="src/shared.py", ac_indices=(0, 1)),
        ]

        review = await coordinator.run_review(
            execution_id="exec_safety_net",
            conflicts=missed_conflicts,
            level_context=level_ctx,
            level_number=1,
        )

        assert review.review_summary == "Resolved missed conflict in shared.py"
        assert review.fixes_applied == ("Merged overlapping edits",)
        assert review.warnings_for_next_level == ("Verify shared.py integration",)
        assert review.conflicts_detected[0].resolved is True
        assert review.session_id == "safety-net-review-1"
