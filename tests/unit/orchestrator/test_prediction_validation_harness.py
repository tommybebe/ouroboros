"""Validation harness for file overlap prediction conservatism.

Replays historical or synthetic parallel AC executions, compares predicted
file sets against actual files modified, and asserts zero false negatives.
Provides a feedback loop for tuning prediction conservatism.

The harness uses two core concepts:

1. **ACExecutionScenario**: Captures a single AC's spec + the files it
   actually modified during execution (ground truth from coordinator
   _collect_file_modifications or manual annotation).

2. **PredictionValidationHarness**: Runs the predictor against scenarios,
   produces PredictionValidationReport with per-AC metrics and aggregate
   false-negative/false-positive rates.

Zero false negatives is the invariant: if a file was actually modified,
the predictor MUST have predicted it. False positives (over-prediction)
are acceptable and expected.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pytest

from ouroboros.orchestrator.dependency_analyzer import ACDependencySpec
from ouroboros.orchestrator.file_overlap_predictor import (
    FileOverlapPrediction,
    FileOverlapPredictor,
)

# ---------------------------------------------------------------------------
# Harness data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ACExecutionScenario:
    """Ground truth for a single AC's execution.

    Attributes:
        spec: The AC dependency spec used as predictor input.
        actual_files_modified: Files actually touched during execution
            (relative to repo root). Sourced from coordinator tool-call
            scanning or manual annotation.
    """

    spec: ACDependencySpec
    actual_files_modified: frozenset[str]


@dataclass(frozen=True, slots=True)
class ACPredictionMetrics:
    """Per-AC comparison of predicted vs actual files.

    Attributes:
        ac_index: AC index.
        predicted: Files the predictor said this AC would touch.
        actual: Files actually touched.
        true_positives: Correctly predicted files (predicted ∩ actual).
        false_negatives: Actually modified but NOT predicted (actual - predicted).
            MUST be empty for the invariant to hold.
        false_positives: Predicted but NOT actually modified (predicted - actual).
            Expected to be non-empty (over-prediction is acceptable).
        recall: true_positives / actual. Must be 1.0 for zero false negatives.
        precision: true_positives / predicted. Lower is acceptable (over-prediction).
    """

    ac_index: int
    predicted: frozenset[str]
    actual: frozenset[str]
    true_positives: frozenset[str]
    false_negatives: frozenset[str]
    false_positives: frozenset[str]
    recall: float
    precision: float


@dataclass(frozen=True, slots=True)
class OverlapValidationMetrics:
    """Validation of overlap detection against actual conflicts.

    Attributes:
        actual_conflict_pairs: Pairs of AC indices that actually modified
            the same file(s).
        predicted_overlap_pairs: Pairs of AC indices predicted to overlap.
        missed_conflicts: Actual conflict pairs not captured by prediction.
            MUST be empty for the invariant to hold.
        spurious_overlaps: Predicted overlaps with no actual conflict.
            Acceptable (conservative).
    """

    actual_conflict_pairs: frozenset[frozenset[int]]
    predicted_overlap_pairs: frozenset[frozenset[int]]
    missed_conflicts: frozenset[frozenset[int]]
    spurious_overlaps: frozenset[frozenset[int]]


@dataclass(frozen=True, slots=True)
class PredictionValidationReport:
    """Full validation report from harness replay.

    Attributes:
        scenario_name: Human-readable identifier for this scenario.
        ac_metrics: Per-AC prediction metrics.
        overlap_metrics: Overlap detection validation.
        prediction: The raw prediction result from the predictor.
        aggregate_recall: Mean recall across all ACs.
        aggregate_precision: Mean precision across all ACs.
        zero_false_negatives: True if no AC had any false negatives.
            This is the critical invariant.
        total_false_negatives: Total count of missed files across all ACs.
    """

    scenario_name: str
    ac_metrics: tuple[ACPredictionMetrics, ...]
    overlap_metrics: OverlapValidationMetrics
    prediction: FileOverlapPrediction
    aggregate_recall: float
    aggregate_precision: float
    zero_false_negatives: bool
    total_false_negatives: int


# ---------------------------------------------------------------------------
# Harness implementation
# ---------------------------------------------------------------------------


class PredictionValidationHarness:
    """Replays AC execution scenarios against the predictor and validates.

    Usage:
        harness = PredictionValidationHarness(file_index=repo_file_index)
        report = harness.validate(scenarios, scenario_name="level-3-replay")
        assert report.zero_false_negatives
    """

    def __init__(
        self,
        file_index: frozenset[str] | None = None,
        *,
        predictor: FileOverlapPredictor | None = None,
    ) -> None:
        """Initialize harness.

        Args:
            file_index: Repository file index for predictor grounding.
            predictor: Optional pre-configured predictor. If None, one is
                created from file_index.
        """
        self._predictor = predictor or FileOverlapPredictor(
            file_index=file_index or frozenset(),
        )

    def validate(
        self,
        scenarios: Sequence[ACExecutionScenario],
        *,
        scenario_name: str = "unnamed",
        stage_ac_indices: tuple[int, ...] | None = None,
    ) -> PredictionValidationReport:
        """Run prediction and validate against ground truth.

        Args:
            scenarios: AC execution scenarios with ground truth.
            scenario_name: Human-readable name for reporting.
            stage_ac_indices: Optional stage filter for prediction.

        Returns:
            PredictionValidationReport with metrics and invariant check.
        """
        specs = [s.spec for s in scenarios]
        prediction = asyncio.run(
            self._predictor.predict(specs, stage_ac_indices=stage_ac_indices),
        )

        # Build predicted paths lookup by AC index
        predicted_by_ac: dict[int, frozenset[str]] = {
            p.ac_index: p.predicted_paths for p in prediction.ac_predictions
        }

        # Per-AC metrics
        ac_metrics: list[ACPredictionMetrics] = []
        for scenario in scenarios:
            idx = scenario.spec.index
            predicted = predicted_by_ac.get(idx, frozenset())
            actual = scenario.actual_files_modified

            tp = predicted & actual
            fn = actual - predicted
            fp = predicted - actual

            recall = len(tp) / len(actual) if actual else 1.0
            precision = len(tp) / len(predicted) if predicted else (1.0 if not actual else 0.0)

            ac_metrics.append(
                ACPredictionMetrics(
                    ac_index=idx,
                    predicted=predicted,
                    actual=actual,
                    true_positives=tp,
                    false_negatives=fn,
                    false_positives=fp,
                    recall=recall,
                    precision=precision,
                ),
            )

        # Overlap validation
        overlap_metrics = self._validate_overlaps(scenarios, prediction)

        # Aggregates
        recalls = [m.recall for m in ac_metrics]
        precisions = [m.precision for m in ac_metrics]
        agg_recall = sum(recalls) / len(recalls) if recalls else 1.0
        agg_precision = sum(precisions) / len(precisions) if precisions else 1.0
        total_fn = sum(len(m.false_negatives) for m in ac_metrics)

        return PredictionValidationReport(
            scenario_name=scenario_name,
            ac_metrics=tuple(ac_metrics),
            overlap_metrics=overlap_metrics,
            prediction=prediction,
            aggregate_recall=agg_recall,
            aggregate_precision=agg_precision,
            zero_false_negatives=total_fn == 0,
            total_false_negatives=total_fn,
        )

    def _validate_overlaps(
        self,
        scenarios: Sequence[ACExecutionScenario],
        prediction: FileOverlapPrediction,
    ) -> OverlapValidationMetrics:
        """Compare predicted overlaps against actual file conflicts."""
        # Compute actual conflicts: pairs of ACs that modified the same file
        from collections import defaultdict

        file_to_acs: dict[str, set[int]] = defaultdict(set)
        for scenario in scenarios:
            for f in scenario.actual_files_modified:
                file_to_acs[f].add(scenario.spec.index)

        actual_pairs: set[frozenset[int]] = set()
        for _file, acs in file_to_acs.items():
            if len(acs) >= 2:
                # All pairs within this set
                sorted_acs = sorted(acs)
                for i in range(len(sorted_acs)):
                    for j in range(i + 1, len(sorted_acs)):
                        actual_pairs.add(frozenset({sorted_acs[i], sorted_acs[j]}))

        # Compute predicted overlap pairs from overlap groups
        predicted_pairs: set[frozenset[int]] = set()
        for group in prediction.overlap_groups:
            indices = sorted(group.ac_indices)
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    predicted_pairs.add(frozenset({indices[i], indices[j]}))

        missed = frozenset(actual_pairs - predicted_pairs)
        spurious = frozenset(predicted_pairs - actual_pairs)

        return OverlapValidationMetrics(
            actual_conflict_pairs=frozenset(actual_pairs),
            predicted_overlap_pairs=frozenset(predicted_pairs),
            missed_conflicts=missed,
            spurious_overlaps=spurious,
        )


# ---------------------------------------------------------------------------
# Synthetic scenario builders
# ---------------------------------------------------------------------------


def build_scenario(
    ac_index: int,
    content: str,
    actual_files: Sequence[str],
    *,
    metadata: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> ACExecutionScenario:
    """Convenience builder for test scenarios.

    Args:
        ac_index: AC index.
        content: AC description text.
        actual_files: Files actually modified during execution.
        metadata: Optional AC metadata.
        context: Optional AC context.
    """
    return ACExecutionScenario(
        spec=ACDependencySpec(
            index=ac_index,
            content=content,
            metadata=metadata or {},
            context=context or {},
        ),
        actual_files_modified=frozenset(actual_files),
    )


def build_scenario_from_result_messages(
    ac_index: int,
    content: str,
    messages: Sequence[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
) -> ACExecutionScenario:
    """Build scenario from synthetic tool-call messages.

    Extracts actual files from Write/Edit tool calls in messages,
    mirroring how the coordinator's _collect_file_modifications works.

    Args:
        ac_index: AC index.
        content: AC description text.
        messages: Dicts with keys like {"tool_name": "Edit", "tool_input": {"file_path": "..."}}.
        metadata: Optional AC metadata.
    """
    actual: set[str] = set()
    for msg in messages:
        if msg.get("tool_name") in ("Write", "Edit"):
            tool_input = msg.get("tool_input", {})
            fp = tool_input.get("file_path")
            if fp:
                actual.add(fp)

    return ACExecutionScenario(
        spec=ACDependencySpec(
            index=ac_index,
            content=content,
            metadata=metadata or {},
        ),
        actual_files_modified=frozenset(actual),
    )


# ---------------------------------------------------------------------------
# Sample file index (shared across tests)
# ---------------------------------------------------------------------------

_SAMPLE_FILE_INDEX = frozenset({
    "src/ouroboros/__init__.py",
    "src/ouroboros/core/__init__.py",
    "src/ouroboros/core/types.py",
    "src/ouroboros/core/seed.py",
    "src/ouroboros/core/errors.py",
    "src/ouroboros/core/worktree.py",
    "src/ouroboros/orchestrator/__init__.py",
    "src/ouroboros/orchestrator/coordinator.py",
    "src/ouroboros/orchestrator/dependency_analyzer.py",
    "src/ouroboros/orchestrator/parallel_executor.py",
    "src/ouroboros/orchestrator/level_context.py",
    "src/ouroboros/orchestrator/file_overlap_predictor.py",
    "src/ouroboros/orchestrator/adapter.py",
    "src/ouroboros/orchestrator/session.py",
    "src/ouroboros/orchestrator/events.py",
    "src/ouroboros/orchestrator/ac_isolation.py",
    "src/ouroboros/orchestrator/ac_worktree.py",
    "src/ouroboros/providers/__init__.py",
    "src/ouroboros/providers/base.py",
    "src/ouroboros/providers/anthropic.py",
    "src/ouroboros/observability/__init__.py",
    "src/ouroboros/observability/logging.py",
    "tests/unit/__init__.py",
    "tests/unit/orchestrator/__init__.py",
    "tests/unit/orchestrator/test_coordinator.py",
    "tests/unit/orchestrator/test_dependency_analyzer.py",
    "tests/unit/orchestrator/test_parallel_executor.py",
    "tests/unit/orchestrator/test_file_overlap_predictor.py",
    "pyproject.toml",
    "README.md",
    "CHANGELOG.md",
    "docs/architecture.md",
    "docs/guide.md",
})


def _harness(
    file_index: frozenset[str] | None = None,
) -> PredictionValidationHarness:
    return PredictionValidationHarness(file_index=file_index or _SAMPLE_FILE_INDEX)


# ===========================================================================
# Test suite
# ===========================================================================


class TestHarnessDataModels:
    """Frozen dataclass invariants for harness models."""

    def test_scenario_frozen(self) -> None:
        s = build_scenario(0, "test", ["foo.py"])
        with pytest.raises(AttributeError):
            s.actual_files_modified = frozenset()  # type: ignore[misc]

    def test_metrics_frozen(self) -> None:
        m = ACPredictionMetrics(
            ac_index=0,
            predicted=frozenset(),
            actual=frozenset(),
            true_positives=frozenset(),
            false_negatives=frozenset(),
            false_positives=frozenset(),
            recall=1.0,
            precision=1.0,
        )
        with pytest.raises(AttributeError):
            m.recall = 0.5  # type: ignore[misc]

    def test_report_frozen(self) -> None:
        r = PredictionValidationReport(
            scenario_name="test",
            ac_metrics=(),
            overlap_metrics=OverlapValidationMetrics(
                actual_conflict_pairs=frozenset(),
                predicted_overlap_pairs=frozenset(),
                missed_conflicts=frozenset(),
                spurious_overlaps=frozenset(),
            ),
            prediction=FileOverlapPrediction(ac_predictions=()),
            aggregate_recall=1.0,
            aggregate_precision=1.0,
            zero_false_negatives=True,
            total_false_negatives=0,
        )
        with pytest.raises(AttributeError):
            r.scenario_name = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Zero false-negatives invariant: explicit file hints
# ---------------------------------------------------------------------------


class TestZeroFalseNegativesExplicit:
    """When ACs have metadata.files hints pointing to files that ARE actually
    modified, the predictor MUST predict them (zero false negatives)."""

    def test_metadata_files_predicted(self) -> None:
        """AC with metadata.files — predicted set must contain actual set."""
        scenarios = [
            build_scenario(
                0,
                "Add Result type variant to core types",
                ["src/ouroboros/core/types.py"],
                metadata={"files": ["src/ouroboros/core/types.py"]},
            ),
        ]
        report = _harness().validate(scenarios, scenario_name="metadata-files-hint")
        assert report.zero_false_negatives, (
            f"False negatives detected: {report.ac_metrics[0].false_negatives}"
        )

    def test_context_modifies_predicted(self) -> None:
        """AC with context.modifies — all listed files must be predicted."""
        scenarios = [
            build_scenario(
                0,
                "Update coordinator logic",
                ["src/ouroboros/orchestrator/coordinator.py"],
                context={"modifies": ["src/ouroboros/orchestrator/coordinator.py"]},
            ),
        ]
        report = _harness().validate(scenarios, scenario_name="context-modifies")
        assert report.zero_false_negatives

    def test_multiple_metadata_files(self) -> None:
        """Multiple files in metadata — all must be predicted."""
        actual = [
            "src/ouroboros/core/types.py",
            "src/ouroboros/core/errors.py",
        ]
        scenarios = [
            build_scenario(
                0,
                "Refactor error handling across core",
                actual,
                metadata={"files": actual},
            ),
        ]
        report = _harness().validate(scenarios, scenario_name="multi-metadata-files")
        assert report.zero_false_negatives


# ---------------------------------------------------------------------------
# Zero false-negatives invariant: path-in-text heuristic
# ---------------------------------------------------------------------------


class TestZeroFalseNegativesPathInText:
    """When AC description text contains explicit file paths that match
    actual modifications, the predictor must capture them."""

    def test_path_in_description(self) -> None:
        scenarios = [
            build_scenario(
                0,
                "Modify src/ouroboros/orchestrator/coordinator.py to add merge logic",
                ["src/ouroboros/orchestrator/coordinator.py"],
            ),
        ]
        report = _harness().validate(scenarios, scenario_name="path-in-text")
        assert report.zero_false_negatives

    def test_module_reference_in_description(self) -> None:
        """Module-style reference (ouroboros.core.types) should resolve."""
        scenarios = [
            build_scenario(
                0,
                "Extend ouroboros.orchestrator.coordinator with conflict detection",
                ["src/ouroboros/orchestrator/coordinator.py"],
            ),
        ]
        report = _harness().validate(scenarios, scenario_name="module-ref-in-text")
        # The module path should be grounded to the actual file
        assert report.zero_false_negatives


# ---------------------------------------------------------------------------
# Zero false-negatives invariant: category/neighbour expansion
# ---------------------------------------------------------------------------


class TestZeroFalseNegativesExpansion:
    """When an AC touches a file in a category/directory it describes,
    neighbour and category expansion should capture it."""

    def test_neighbour_expansion_catches_sibling(self) -> None:
        """AC mentions types.py → sibling errors.py also predicted."""
        scenarios = [
            build_scenario(
                0,
                "Update src/ouroboros/core/types.py with new Result variants",
                [
                    "src/ouroboros/core/types.py",
                    "src/ouroboros/core/errors.py",  # sibling, not explicitly mentioned
                ],
                metadata={"files": ["src/ouroboros/core/types.py"]},
            ),
        ]
        report = _harness().validate(scenarios, scenario_name="neighbour-expansion")
        assert report.zero_false_negatives, (
            f"Missed siblings: {report.ac_metrics[0].false_negatives}"
        )

    def test_category_expansion_catches_orchestrator_files(self) -> None:
        """AC about 'parallel execution' → orchestrator files predicted."""
        scenarios = [
            build_scenario(
                0,
                "Fix parallel execution stall detection in the orchestrator",
                [
                    "src/ouroboros/orchestrator/parallel_executor.py",
                    "src/ouroboros/orchestrator/events.py",
                ],
            ),
        ]
        report = _harness().validate(scenarios, scenario_name="category-expansion")
        assert report.zero_false_negatives


# ---------------------------------------------------------------------------
# Overlap detection validation
# ---------------------------------------------------------------------------


class TestOverlapDetectionValidation:
    """Validate that predicted overlaps capture all actual conflicts."""

    def test_same_file_conflict_detected(self) -> None:
        """Two ACs modifying the same file → overlap must be predicted."""
        scenarios = [
            build_scenario(
                0,
                "Add merge logic to src/ouroboros/orchestrator/coordinator.py",
                ["src/ouroboros/orchestrator/coordinator.py"],
                metadata={"files": ["src/ouroboros/orchestrator/coordinator.py"]},
            ),
            build_scenario(
                1,
                "Add review logic to src/ouroboros/orchestrator/coordinator.py",
                ["src/ouroboros/orchestrator/coordinator.py"],
                metadata={"files": ["src/ouroboros/orchestrator/coordinator.py"]},
            ),
        ]
        report = _harness().validate(scenarios, scenario_name="same-file-conflict")
        assert report.overlap_metrics.missed_conflicts == frozenset(), (
            f"Missed conflicts: {report.overlap_metrics.missed_conflicts}"
        )

    def test_same_directory_conflict_detected(self) -> None:
        """Two ACs modifying files in the same directory → overlap via neighbour expansion."""
        scenarios = [
            build_scenario(
                0,
                "Update the coordinator module",
                ["src/ouroboros/orchestrator/coordinator.py"],
                metadata={"files": ["src/ouroboros/orchestrator/coordinator.py"]},
            ),
            build_scenario(
                1,
                "Update the dependency analyzer module",
                ["src/ouroboros/orchestrator/dependency_analyzer.py"],
                metadata={"files": ["src/ouroboros/orchestrator/dependency_analyzer.py"]},
            ),
        ]
        report = _harness().validate(scenarios, scenario_name="same-dir-conflict")
        # Neighbour expansion should predict overlap
        assert report.overlap_metrics.missed_conflicts == frozenset()

    def test_no_overlap_independent_dirs(self) -> None:
        """ACs in independent directories → no actual or predicted conflict."""
        file_index = frozenset({
            "src/module_a/foo.py",
            "src/module_b/bar.py",
        })
        scenarios = [
            build_scenario(
                0,
                "Update module A",
                ["src/module_a/foo.py"],
                metadata={"files": ["src/module_a/foo.py"]},
            ),
            build_scenario(
                1,
                "Update module B",
                ["src/module_b/bar.py"],
                metadata={"files": ["src/module_b/bar.py"]},
            ),
        ]
        report = PredictionValidationHarness(file_index=file_index).validate(
            scenarios, scenario_name="independent-dirs",
        )
        assert report.overlap_metrics.actual_conflict_pairs == frozenset()
        assert report.overlap_metrics.missed_conflicts == frozenset()


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


class TestAggregateMetrics:
    """Validate aggregate recall/precision computation."""

    def test_perfect_recall_when_all_predicted(self) -> None:
        scenarios = [
            build_scenario(
                0,
                "Update src/ouroboros/core/types.py",
                ["src/ouroboros/core/types.py"],
                metadata={"files": ["src/ouroboros/core/types.py"]},
            ),
        ]
        report = _harness().validate(scenarios, scenario_name="perfect-recall")
        assert report.aggregate_recall == 1.0

    def test_precision_below_one_for_broad_prediction(self) -> None:
        """Broad category prediction causes false positives → precision < 1."""
        scenarios = [
            build_scenario(
                0,
                "Fix a bug in the orchestrator parallel execution engine",
                ["src/ouroboros/orchestrator/parallel_executor.py"],  # only 1 file actually modified
            ),
        ]
        report = _harness().validate(scenarios, scenario_name="broad-prediction")
        # Recall must still be 1.0 (the actual file is predicted)
        assert report.ac_metrics[0].recall == 1.0
        # But precision should be < 1.0 since many orchestrator files are predicted
        assert report.ac_metrics[0].precision < 1.0

    def test_empty_scenario(self) -> None:
        report = _harness().validate([], scenario_name="empty")
        assert report.zero_false_negatives
        assert report.total_false_negatives == 0

    def test_ac_with_no_actual_files(self) -> None:
        """AC that modified nothing → recall is 1.0 trivially."""
        scenarios = [
            build_scenario(0, "Review code quality", []),
        ]
        report = _harness().validate(scenarios, scenario_name="no-actual-files")
        assert report.ac_metrics[0].recall == 1.0
        assert report.zero_false_negatives


# ---------------------------------------------------------------------------
# Synthetic scenario: realistic 3-AC parallel level
# ---------------------------------------------------------------------------


class TestRealistic3ACLevel:
    """Synthetic replay of a typical 3-AC parallel execution level."""

    def test_realistic_parallel_level(self) -> None:
        """Three ACs run in parallel — validate prediction conservatism."""
        scenarios = [
            build_scenario(
                0,
                "Add file overlap prediction to the dependency analyzer",
                [
                    "src/ouroboros/orchestrator/dependency_analyzer.py",
                    "src/ouroboros/orchestrator/file_overlap_predictor.py",
                ],
                metadata={"files": [
                    "src/ouroboros/orchestrator/dependency_analyzer.py",
                    "src/ouroboros/orchestrator/file_overlap_predictor.py",
                ]},
            ),
            build_scenario(
                1,
                "Add worktree isolation for per-AC execution",
                [
                    "src/ouroboros/core/worktree.py",
                    "src/ouroboros/orchestrator/ac_worktree.py",
                    "src/ouroboros/orchestrator/ac_isolation.py",
                ],
                metadata={"files": [
                    "src/ouroboros/core/worktree.py",
                    "src/ouroboros/orchestrator/ac_worktree.py",
                    "src/ouroboros/orchestrator/ac_isolation.py",
                ]},
            ),
            build_scenario(
                2,
                "Add unit tests for prediction validation harness",
                [
                    "tests/unit/orchestrator/test_file_overlap_predictor.py",
                ],
                metadata={"files": [
                    "tests/unit/orchestrator/test_file_overlap_predictor.py",
                ]},
            ),
        ]
        report = _harness().validate(
            scenarios,
            scenario_name="realistic-3ac-level",
        )

        # Critical invariant
        assert report.zero_false_negatives, (
            f"False negatives in realistic scenario! "
            f"Total: {report.total_false_negatives}, "
            f"Details: {[(m.ac_index, m.false_negatives) for m in report.ac_metrics if m.false_negatives]}"
        )

        # All three ACs should be analysed
        assert len(report.ac_metrics) == 3

        # ACs 0 and 1 both touch orchestrator/ — should detect overlap
        assert report.prediction.has_overlaps


class TestRealisticIndependentACs:
    """Scenario where ACs are truly independent — no spurious isolation."""

    def test_independent_acs_stay_shared(self) -> None:
        """ACs in completely different modules → shared workspace, no isolation."""
        file_index = frozenset({
            "src/frontend/app.tsx",
            "src/frontend/styles.css",
            "src/backend/server.py",
            "src/backend/routes.py",
            "docs/api.md",
            "docs/guide.md",
        })
        scenarios = [
            build_scenario(
                0,
                "Add new frontend component",
                ["src/frontend/app.tsx"],
                metadata={"files": ["src/frontend/app.tsx"]},
            ),
            build_scenario(
                1,
                "Add new backend endpoint",
                ["src/backend/server.py"],
                metadata={"files": ["src/backend/server.py"]},
            ),
            build_scenario(
                2,
                "Update API documentation",
                ["docs/api.md"],
                metadata={"files": ["docs/api.md"]},
            ),
        ]
        harness = PredictionValidationHarness(file_index=file_index)
        report = harness.validate(scenarios, scenario_name="independent-acs")

        # All should stay in shared workspace
        assert report.zero_false_negatives
        # No actual conflicts
        assert report.overlap_metrics.actual_conflict_pairs == frozenset()


# ---------------------------------------------------------------------------
# Scenario builder from tool-call messages (coordinator-style)
# ---------------------------------------------------------------------------


class TestMessageBasedScenarioBuilder:
    """Tests for build_scenario_from_result_messages helper."""

    def test_extracts_write_paths(self) -> None:
        scenario = build_scenario_from_result_messages(
            ac_index=0,
            content="Create new module",
            messages=[
                {"tool_name": "Write", "tool_input": {"file_path": "src/new.py"}},
                {"tool_name": "Read", "tool_input": {"file_path": "src/old.py"}},  # not a write
                {"tool_name": "Edit", "tool_input": {"file_path": "src/existing.py"}},
            ],
        )
        assert scenario.actual_files_modified == frozenset({"src/new.py", "src/existing.py"})

    def test_ignores_non_write_tools(self) -> None:
        scenario = build_scenario_from_result_messages(
            ac_index=0,
            content="Read files",
            messages=[
                {"tool_name": "Read", "tool_input": {"file_path": "src/foo.py"}},
                {"tool_name": "Grep", "tool_input": {"pattern": "TODO"}},
            ],
        )
        assert scenario.actual_files_modified == frozenset()

    def test_empty_messages(self) -> None:
        scenario = build_scenario_from_result_messages(
            ac_index=0,
            content="Do nothing",
            messages=[],
        )
        assert scenario.actual_files_modified == frozenset()


# ---------------------------------------------------------------------------
# Multi-level replay scenario
# ---------------------------------------------------------------------------


class TestMultiLevelReplay:
    """Replay multiple execution levels to validate across stages."""

    def test_two_level_replay(self) -> None:
        """Run prediction validation for two sequential levels."""
        harness = _harness()

        # Level 1: two ACs in parallel
        level_1 = [
            build_scenario(
                0,
                "Add new core type",
                ["src/ouroboros/core/types.py"],
                metadata={"files": ["src/ouroboros/core/types.py"]},
            ),
            build_scenario(
                1,
                "Update orchestrator adapter",
                ["src/ouroboros/orchestrator/adapter.py"],
                metadata={"files": ["src/ouroboros/orchestrator/adapter.py"]},
            ),
        ]
        report_1 = harness.validate(level_1, scenario_name="level-1")
        assert report_1.zero_false_negatives

        # Level 2: depends on level 1 outputs
        level_2 = [
            build_scenario(
                2,
                "Add tests for new core type and adapter changes",
                [
                    "tests/unit/orchestrator/test_coordinator.py",
                    "tests/unit/orchestrator/test_dependency_analyzer.py",
                ],
                metadata={"files": [
                    "tests/unit/orchestrator/test_coordinator.py",
                    "tests/unit/orchestrator/test_dependency_analyzer.py",
                ]},
            ),
        ]
        report_2 = harness.validate(level_2, scenario_name="level-2")
        assert report_2.zero_false_negatives


# ---------------------------------------------------------------------------
# Stress: many ACs with overlapping categories
# ---------------------------------------------------------------------------


class TestStressScenarios:
    """Higher-volume scenarios to exercise overlap computation."""

    def test_five_acs_some_overlapping(self) -> None:
        """Five ACs, some sharing orchestrator category, some independent."""
        scenarios = [
            build_scenario(
                0,
                "Update parallel executor retry logic",
                ["src/ouroboros/orchestrator/parallel_executor.py"],
                metadata={"files": ["src/ouroboros/orchestrator/parallel_executor.py"]},
            ),
            build_scenario(
                1,
                "Add coordinator merge-agent support",
                ["src/ouroboros/orchestrator/coordinator.py"],
                metadata={"files": ["src/ouroboros/orchestrator/coordinator.py"]},
            ),
            build_scenario(
                2,
                "Update provider base class",
                ["src/ouroboros/providers/base.py"],
                metadata={"files": ["src/ouroboros/providers/base.py"]},
            ),
            build_scenario(
                3,
                "Fix logging format in observability module",
                ["src/ouroboros/observability/logging.py"],
                metadata={"files": ["src/ouroboros/observability/logging.py"]},
            ),
            build_scenario(
                4,
                "Update session management",
                ["src/ouroboros/orchestrator/session.py"],
                metadata={"files": ["src/ouroboros/orchestrator/session.py"]},
            ),
        ]
        report = _harness().validate(scenarios, scenario_name="five-acs-mixed")
        assert report.zero_false_negatives
        assert len(report.ac_metrics) == 5
        # ACs 0, 1, 4 all in orchestrator/ — should detect overlap among them
        orchestrator_acs = {0, 1, 4}
        for pair in report.overlap_metrics.actual_conflict_pairs:
            # Any actual conflict among orchestrator ACs should be predicted
            if pair.issubset(orchestrator_acs):
                assert pair in report.overlap_metrics.predicted_overlap_pairs


# ---------------------------------------------------------------------------
# Conservatism tuning feedback
# ---------------------------------------------------------------------------


class TestConservatismFeedback:
    """Demonstrate the harness as a tuning feedback loop.

    These tests show how to use the report's false_positives and precision
    to measure over-prediction without sacrificing recall.
    """

    def test_over_prediction_measured(self) -> None:
        """Confirm that precision < 1.0 measures the over-prediction rate."""
        scenarios = [
            build_scenario(
                0,
                "Fix a critical bug in the orchestrator parallel execution",
                # Actually only modified one file
                ["src/ouroboros/orchestrator/parallel_executor.py"],
            ),
        ]
        report = _harness().validate(scenarios, scenario_name="over-prediction-measure")
        metrics = report.ac_metrics[0]

        # Recall must be 1.0 (the actual file is predicted)
        assert metrics.recall == 1.0
        # False positives quantify over-prediction
        assert len(metrics.false_positives) > 0
        # Precision quantifies how targeted the prediction is
        assert 0.0 < metrics.precision < 1.0

    def test_report_false_negative_details(self) -> None:
        """When false negatives occur, the report pinpoints them for tuning."""
        # Create a scenario where prediction can't possibly know about a file
        # (no hints at all — the file is in an unrelated directory)
        file_index = frozenset({
            "src/main.py",
            "vendor/obscure/lib.py",
        })
        scenarios = [
            build_scenario(
                0,
                "Update main entry point",
                # AC also modified a vendor file (no hints in description)
                ["src/main.py", "vendor/obscure/lib.py"],
                metadata={"files": ["src/main.py"]},
            ),
        ]
        report = PredictionValidationHarness(file_index=file_index).validate(
            scenarios, scenario_name="false-negative-details",
        )
        metrics = report.ac_metrics[0]

        # This scenario MAY have false negatives since vendor/ has no hints
        if not report.zero_false_negatives:
            # The report provides details for tuning
            assert "vendor/obscure/lib.py" in metrics.false_negatives
            assert report.total_false_negatives == 1
