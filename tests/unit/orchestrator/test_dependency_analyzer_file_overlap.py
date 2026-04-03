"""Tests for DependencyAnalyzer file overlap prediction at planning time.

Verifies that analyze_with_file_overlap() integrates the FileOverlapPredictor
into the dependency analysis pipeline so file overlap is predicted *before*
execution begins rather than per-level at execution time.
"""

from __future__ import annotations

from typing import Any

import pytest

from ouroboros.core.errors import ProviderError
from ouroboros.core.types import Result
from ouroboros.orchestrator.dependency_analyzer import (
    ACDependencySpec,
    ACNode,
    DependencyAnalyzer,
    DependencyGraph,
    StagedExecutionPlan,
)
from ouroboros.orchestrator.file_overlap_predictor import (
    ACFilePrediction,
    FileOverlapPrediction,
    OverlapGroup,
)
from ouroboros.providers.base import CompletionResponse, UsageInfo


class StubLLMAdapter:
    """Minimal LLM stub returning independent ACs."""

    def __init__(self, content: str | None = None) -> None:
        self._content = content

    async def complete(
        self, messages: list[Any], config: Any
    ) -> Result[CompletionResponse, ProviderError]:
        return Result.ok(
            CompletionResponse(
                content=self._content or '{"dependencies": []}',
                model="test-model",
                usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
        )


def _all_independent_response(ac_count: int) -> str:
    items = ",".join(f'{{"ac_index": {i}, "depends_on": []}}' for i in range(ac_count))
    return f'{{"dependencies": [{items}]}}'


# ---------------------------------------------------------------------------
# analyze_with_file_overlap basics
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_analyze_with_file_overlap_no_repo_root_skips_prediction() -> None:
    """When no repo_root or file_index is provided, prediction is skipped."""
    llm = StubLLMAdapter(_all_independent_response(3))
    analyzer = DependencyAnalyzer(llm_adapter=llm)

    specs = [
        ACDependencySpec(index=0, content="Add tests for module A"),
        ACDependencySpec(index=1, content="Add tests for module B"),
        ACDependencySpec(index=2, content="Add docs for module C"),
    ]
    result = await analyzer.analyze_with_file_overlap(specs)

    assert result.is_ok
    graph = result.value
    # No prediction attached when no repo root
    assert graph.file_overlap_prediction is None


@pytest.mark.anyio
async def test_analyze_with_file_overlap_with_file_index() -> None:
    """When a file_index is provided, prediction runs at planning time."""
    llm = StubLLMAdapter(_all_independent_response(2))
    analyzer = DependencyAnalyzer(llm_adapter=llm)

    # Two ACs that mention the same directory pattern
    specs = [
        ACDependencySpec(
            index=0,
            content="Update the orchestrator coordinator module",
        ),
        ACDependencySpec(
            index=1,
            content="Fix the orchestrator executor module",
        ),
    ]

    file_index = frozenset({
        "src/ouroboros/orchestrator/coordinator.py",
        "src/ouroboros/orchestrator/parallel_executor.py",
        "src/ouroboros/orchestrator/__init__.py",
        "src/ouroboros/core/types.py",
    })

    result = await analyzer.analyze_with_file_overlap(
        specs,
        file_index=file_index,
    )

    assert result.is_ok
    graph = result.value
    # Prediction should be attached since we provided a file_index
    assert graph.file_overlap_prediction is not None


@pytest.mark.anyio
async def test_analyze_with_file_overlap_single_ac_skips_prediction() -> None:
    """Single-AC analysis skips file overlap prediction entirely."""
    analyzer = DependencyAnalyzer()

    specs = [ACDependencySpec(index=0, content="Do something")]
    result = await analyzer.analyze_with_file_overlap(
        specs,
        file_index=frozenset({"src/main.py"}),
    )

    assert result.is_ok
    graph = result.value
    # Single AC means no parallel levels, so prediction is skipped
    assert graph.file_overlap_prediction is None


@pytest.mark.anyio
async def test_analyze_with_file_overlap_serial_only_skips_prediction() -> None:
    """When all ACs are serialized (no parallel levels), prediction is skipped."""
    # AC 1 depends on AC 0 → both in serial stages
    llm = StubLLMAdapter(
        '{"dependencies": [{"ac_index": 0, "depends_on": []}, {"ac_index": 1, "depends_on": [0]}]}'
    )
    analyzer = DependencyAnalyzer(llm_adapter=llm)

    specs = [
        ACDependencySpec(index=0, content="Create the base module"),
        ACDependencySpec(index=1, content="Extend the base module"),
    ]
    result = await analyzer.analyze_with_file_overlap(
        specs,
        file_index=frozenset({"src/base.py"}),
    )

    assert result.is_ok
    graph = result.value
    # Both ACs in serial stages → no parallel levels → no prediction
    assert graph.file_overlap_prediction is None


# ---------------------------------------------------------------------------
# DependencyGraph propagation to StagedExecutionPlan
# ---------------------------------------------------------------------------


def test_graph_with_prediction_propagates_to_execution_plan() -> None:
    """File overlap prediction on DependencyGraph propagates to StagedExecutionPlan."""
    # Build a graph with 3 independent ACs and a mock prediction
    nodes = (
        ACNode(index=0, content="AC 0"),
        ACNode(index=1, content="AC 1"),
        ACNode(index=2, content="AC 2"),
    )
    prediction = FileOverlapPrediction(
        ac_predictions=(
            ACFilePrediction(ac_index=0, predicted_paths=frozenset({"src/a.py", "src/shared.py"})),
            ACFilePrediction(ac_index=1, predicted_paths=frozenset({"src/b.py", "src/shared.py"})),
            ACFilePrediction(ac_index=2, predicted_paths=frozenset({"tests/c.py"})),
        ),
        overlap_groups=(
            OverlapGroup(ac_indices=(0, 1), shared_paths=frozenset({"src/shared.py"})),
        ),
        isolated_ac_indices=frozenset({0, 1}),
        shared_ac_indices=frozenset({2}),
    )
    graph = DependencyGraph(
        nodes=nodes,
        execution_levels=((0, 1, 2),),
        file_overlap_prediction=prediction,
    )

    plan = graph.to_execution_plan()

    assert plan.has_file_overlap_predictions
    # Stage 0 has all 3 ACs (parallel)
    stage_pred = plan.get_stage_prediction(0)
    assert stage_pred is not None
    assert stage_pred.has_overlaps
    assert len(stage_pred.overlap_groups) == 1
    assert stage_pred.overlap_groups[0].ac_indices == (0, 1)
    assert 0 in stage_pred.isolated_ac_indices
    assert 1 in stage_pred.isolated_ac_indices
    assert 2 in stage_pred.shared_ac_indices


def test_graph_without_prediction_has_no_stage_predictions() -> None:
    """StagedExecutionPlan has empty predictions when graph has no prediction."""
    nodes = (
        ACNode(index=0, content="AC 0"),
        ACNode(index=1, content="AC 1"),
    )
    graph = DependencyGraph(
        nodes=nodes,
        execution_levels=((0, 1),),
        file_overlap_prediction=None,
    )

    plan = graph.to_execution_plan()

    assert not plan.has_file_overlap_predictions
    assert plan.get_stage_prediction(0) is None


def test_prediction_sliced_correctly_across_multiple_stages() -> None:
    """Predictions are sliced per stage when graph has multiple stages."""
    # Stage 0: ACs 0, 1 (parallel with overlap)
    # Stage 1: AC 2 depends on 0 (serial, single AC)
    nodes = (
        ACNode(index=0, content="AC 0"),
        ACNode(index=1, content="AC 1"),
        ACNode(index=2, content="AC 2", depends_on=(0,)),
    )
    prediction = FileOverlapPrediction(
        ac_predictions=(
            ACFilePrediction(ac_index=0, predicted_paths=frozenset({"src/x.py"})),
            ACFilePrediction(ac_index=1, predicted_paths=frozenset({"src/x.py"})),
            ACFilePrediction(ac_index=2, predicted_paths=frozenset({"src/y.py"})),
        ),
        overlap_groups=(
            OverlapGroup(ac_indices=(0, 1), shared_paths=frozenset({"src/x.py"})),
        ),
        isolated_ac_indices=frozenset({0, 1}),
        shared_ac_indices=frozenset({2}),
    )
    graph = DependencyGraph(
        nodes=nodes,
        execution_levels=((0, 1), (2,)),
        file_overlap_prediction=prediction,
    )

    plan = graph.to_execution_plan()

    # Stage 0 (parallel) should have the overlap prediction
    stage0_pred = plan.get_stage_prediction(0)
    assert stage0_pred is not None
    assert stage0_pred.has_overlaps
    assert len(stage0_pred.overlap_groups) == 1

    # Stage 1 (single AC) should have no prediction (not parallel)
    stage1_pred = plan.get_stage_prediction(1)
    assert stage1_pred is None


def test_overlap_group_filtered_to_stage_members() -> None:
    """Overlap groups only include ACs that are in the current stage."""
    # Graph-level prediction has group {0, 1, 3} but stage 0 only has {0, 1}
    nodes = (
        ACNode(index=0, content="AC 0"),
        ACNode(index=1, content="AC 1"),
        ACNode(index=2, content="AC 2"),
        ACNode(index=3, content="AC 3", depends_on=(0,)),
    )
    prediction = FileOverlapPrediction(
        ac_predictions=(
            ACFilePrediction(ac_index=0, predicted_paths=frozenset({"src/x.py"})),
            ACFilePrediction(ac_index=1, predicted_paths=frozenset({"src/x.py"})),
            ACFilePrediction(ac_index=2, predicted_paths=frozenset({"src/y.py"})),
            ACFilePrediction(ac_index=3, predicted_paths=frozenset({"src/x.py"})),
        ),
        overlap_groups=(
            OverlapGroup(ac_indices=(0, 1, 3), shared_paths=frozenset({"src/x.py"})),
        ),
        isolated_ac_indices=frozenset({0, 1, 3}),
        shared_ac_indices=frozenset({2}),
    )
    graph = DependencyGraph(
        nodes=nodes,
        execution_levels=((0, 1, 2), (3,)),
        file_overlap_prediction=prediction,
    )

    plan = graph.to_execution_plan()

    # Stage 0 should filter the overlap group to only {0, 1}
    stage0_pred = plan.get_stage_prediction(0)
    assert stage0_pred is not None
    assert len(stage0_pred.overlap_groups) == 1
    assert stage0_pred.overlap_groups[0].ac_indices == (0, 1)
    # AC 2 is shared (not in any overlap group for this stage)
    assert 2 in stage0_pred.shared_ac_indices


# ---------------------------------------------------------------------------
# StagedExecutionPlan property tests
# ---------------------------------------------------------------------------


def test_staged_execution_plan_get_stage_prediction_missing() -> None:
    """get_stage_prediction returns None for non-existent stage indices."""
    plan = StagedExecutionPlan(nodes=(), stages=())
    assert plan.get_stage_prediction(0) is None
    assert plan.get_stage_prediction(99) is None


def test_staged_execution_plan_has_file_overlap_predictions_empty() -> None:
    """has_file_overlap_predictions is False when dict is empty."""
    plan = StagedExecutionPlan(nodes=(), stages=())
    assert not plan.has_file_overlap_predictions


def test_staged_execution_plan_has_file_overlap_predictions_populated() -> None:
    """has_file_overlap_predictions is True when predictions are present."""
    pred = FileOverlapPrediction(ac_predictions=())
    plan = StagedExecutionPlan(
        nodes=(),
        stages=(),
        file_overlap_predictions={0: pred},
    )
    assert plan.has_file_overlap_predictions


# ---------------------------------------------------------------------------
# DependencyGraph field tests
# ---------------------------------------------------------------------------


def test_dependency_graph_default_prediction_is_none() -> None:
    """DependencyGraph has no prediction by default."""
    graph = DependencyGraph(nodes=())
    assert graph.file_overlap_prediction is None


def test_dependency_graph_carries_prediction() -> None:
    """DependencyGraph can carry a file overlap prediction."""
    pred = FileOverlapPrediction(ac_predictions=())
    graph = DependencyGraph(nodes=(), file_overlap_prediction=pred)
    assert graph.file_overlap_prediction is pred


# ---------------------------------------------------------------------------
# Error resilience
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_prediction_failure_does_not_block_analysis() -> None:
    """If file overlap prediction fails, analysis still succeeds without prediction."""
    llm = StubLLMAdapter(_all_independent_response(2))
    analyzer = DependencyAnalyzer(llm_adapter=llm)

    specs = [
        ACDependencySpec(index=0, content="AC zero"),
        ACDependencySpec(index=1, content="AC one"),
    ]

    # Use a file_index that will allow prediction to run,
    # but the predictor should gracefully handle edge cases
    result = await analyzer.analyze_with_file_overlap(
        specs,
        file_index=frozenset(),
    )

    assert result.is_ok
    graph = result.value
    # Even with empty file index, should produce a prediction (possibly trivial)
    # The important thing is analysis doesn't fail
    assert len(graph.nodes) == 2
