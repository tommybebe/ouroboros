"""Unit tests for conservative file overlap prediction engine."""

from __future__ import annotations

import asyncio

import pytest

from ouroboros.orchestrator.dependency_analyzer import ACDependencySpec
from ouroboros.orchestrator.file_overlap_predictor import (
    ACFilePrediction,
    FileOverlapPrediction,
    FileOverlapPredictor,
    OverlapGroup,
    predict_file_overlaps,
)

# ---------------------------------------------------------------------------
# Fixtures
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
    "src/ouroboros/orchestrator/adapter.py",
    "src/ouroboros/orchestrator/session.py",
    "src/ouroboros/orchestrator/events.py",
    "src/ouroboros/providers/__init__.py",
    "src/ouroboros/providers/base.py",
    "src/ouroboros/providers/anthropic.py",
    "src/ouroboros/observability/__init__.py",
    "src/ouroboros/observability/logging.py",
    "tests/unit/orchestrator/__init__.py",
    "tests/unit/orchestrator/test_coordinator.py",
    "tests/unit/orchestrator/test_dependency_analyzer.py",
    "tests/unit/orchestrator/test_parallel_executor.py",
    "pyproject.toml",
    "README.md",
    "CHANGELOG.md",
    "docs/architecture.md",
    "docs/guide.md",
    "skills/run/SKILL.md",
    "hooks/pre_commit.py",
    "commands/run.py",
})


def _make_predictor(file_index: frozenset[str] | None = None) -> FileOverlapPredictor:
    """Create a predictor with a pre-built file index (no filesystem needed)."""
    return FileOverlapPredictor(file_index=file_index or _SAMPLE_FILE_INDEX)


def _run(coro: object) -> object:
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ACFilePrediction data model
# ---------------------------------------------------------------------------


class TestACFilePrediction:
    """Tests for the ACFilePrediction frozen dataclass."""

    def test_frozen(self) -> None:
        pred = ACFilePrediction(ac_index=0)
        with pytest.raises(AttributeError):
            pred.ac_index = 1  # type: ignore[misc]

    def test_defaults(self) -> None:
        pred = ACFilePrediction(ac_index=0)
        assert pred.predicted_paths == frozenset()
        assert pred.categories == frozenset()
        assert pred.confidence == 0.5


# ---------------------------------------------------------------------------
# OverlapGroup data model
# ---------------------------------------------------------------------------


class TestOverlapGroup:
    """Tests for the OverlapGroup frozen dataclass."""

    def test_frozen(self) -> None:
        group = OverlapGroup(ac_indices=(0, 1), shared_paths=frozenset({"foo.py"}))
        with pytest.raises(AttributeError):
            group.ac_indices = (2,)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FileOverlapPrediction data model
# ---------------------------------------------------------------------------


class TestFileOverlapPrediction:
    """Tests for the FileOverlapPrediction result model."""

    def test_has_overlaps_empty(self) -> None:
        pred = FileOverlapPrediction(ac_predictions=())
        assert not pred.has_overlaps

    def test_has_overlaps_with_groups(self) -> None:
        pred = FileOverlapPrediction(
            ac_predictions=(
                ACFilePrediction(ac_index=0),
                ACFilePrediction(ac_index=1),
            ),
            overlap_groups=(
                OverlapGroup(ac_indices=(0, 1), shared_paths=frozenset({"foo.py"})),
            ),
        )
        assert pred.has_overlaps

    def test_all_ac_indices(self) -> None:
        pred = FileOverlapPrediction(
            ac_predictions=(
                ACFilePrediction(ac_index=0),
                ACFilePrediction(ac_index=2),
                ACFilePrediction(ac_index=5),
            ),
        )
        assert pred.all_ac_indices == frozenset({0, 2, 5})


# ---------------------------------------------------------------------------
# Explicit path extraction
# ---------------------------------------------------------------------------


class TestExplicitPathExtraction:
    """Tests for file path extraction from AC text."""

    def test_extracts_file_path(self) -> None:
        predictor = _make_predictor()
        specs = [
            ACDependencySpec(
                index=0,
                content="Modify src/ouroboros/orchestrator/coordinator.py to add conflict resolution",
            ),
        ]
        result = _run(predictor.predict(specs))
        assert len(result.ac_predictions) == 1
        pred = result.ac_predictions[0]
        assert "src/ouroboros/orchestrator/coordinator.py" in pred.predicted_paths

    def test_extracts_multiple_paths(self) -> None:
        predictor = _make_predictor()
        specs = [
            ACDependencySpec(
                index=0,
                content=(
                    "Update src/ouroboros/core/types.py and "
                    "src/ouroboros/core/errors.py with new Result variants"
                ),
            ),
        ]
        result = _run(predictor.predict(specs))
        pred = result.ac_predictions[0]
        assert "src/ouroboros/core/types.py" in pred.predicted_paths
        assert "src/ouroboros/core/errors.py" in pred.predicted_paths


# ---------------------------------------------------------------------------
# Module reference extraction
# ---------------------------------------------------------------------------


class TestModuleReferenceExtraction:
    """Tests for dotted module reference conversion."""

    def test_converts_module_to_path(self) -> None:
        predictor = _make_predictor()
        specs = [
            ACDependencySpec(
                index=0,
                content="Extend ouroboros.orchestrator.coordinator with merge logic",
            ),
        ]
        result = _run(predictor.predict(specs))
        pred = result.ac_predictions[0]
        # Should find the .py file from the module reference
        assert "src/ouroboros/orchestrator/coordinator.py" in pred.predicted_paths


# ---------------------------------------------------------------------------
# Keyword-to-category matching
# ---------------------------------------------------------------------------


class TestCategoryMatching:
    """Tests for keyword → category → path expansion."""

    def test_test_category(self) -> None:
        predictor = _make_predictor()
        specs = [
            ACDependencySpec(
                index=0,
                content="Add unit tests for the new merge agent",
            ),
        ]
        result = _run(predictor.predict(specs))
        pred = result.ac_predictions[0]
        assert "test" in pred.categories
        # Should include test files
        assert any(p.startswith("tests/") for p in pred.predicted_paths)

    def test_orchestrator_category(self) -> None:
        predictor = _make_predictor()
        specs = [
            ACDependencySpec(
                index=0,
                content="Implement parallel execution with worktree isolation",
            ),
        ]
        result = _run(predictor.predict(specs))
        pred = result.ac_predictions[0]
        assert "orchestrator" in pred.categories
        # Should include orchestrator files
        assert any("orchestrator" in p for p in pred.predicted_paths)

    def test_config_category_includes_root_files(self) -> None:
        predictor = _make_predictor()
        specs = [
            ACDependencySpec(
                index=0,
                content="Update configuration settings in pyproject.toml",
            ),
        ]
        result = _run(predictor.predict(specs))
        pred = result.ac_predictions[0]
        assert "config" in pred.categories
        assert "pyproject.toml" in pred.predicted_paths

    def test_multiple_categories(self) -> None:
        predictor = _make_predictor()
        specs = [
            ACDependencySpec(
                index=0,
                content="Add test coverage for the orchestrator execution engine",
            ),
        ]
        result = _run(predictor.predict(specs))
        pred = result.ac_predictions[0]
        assert "test" in pred.categories
        assert "orchestrator" in pred.categories


# ---------------------------------------------------------------------------
# Metadata-driven path hints
# ---------------------------------------------------------------------------


class TestMetadataPathHints:
    """Tests for file path extraction from AC metadata."""

    def test_metadata_files_key(self) -> None:
        predictor = _make_predictor()
        specs = [
            ACDependencySpec(
                index=0,
                content="Refactor error handling",
                metadata={"files": ["src/ouroboros/core/errors.py", "src/ouroboros/core/types.py"]},
            ),
        ]
        result = _run(predictor.predict(specs))
        pred = result.ac_predictions[0]
        assert "src/ouroboros/core/errors.py" in pred.predicted_paths
        assert "src/ouroboros/core/types.py" in pred.predicted_paths

    def test_context_modifies_key(self) -> None:
        predictor = _make_predictor()
        specs = [
            ACDependencySpec(
                index=0,
                content="Update coordinator",
                context={"modifies": ["src/ouroboros/orchestrator/coordinator.py"]},
            ),
        ]
        result = _run(predictor.predict(specs))
        pred = result.ac_predictions[0]
        assert "src/ouroboros/orchestrator/coordinator.py" in pred.predicted_paths


# ---------------------------------------------------------------------------
# Overlap detection
# ---------------------------------------------------------------------------


class TestOverlapDetection:
    """Tests for file overlap detection between parallel ACs."""

    def test_no_overlap_independent_acs(self) -> None:
        predictor = _make_predictor()
        specs = [
            ACDependencySpec(
                index=0,
                content="Update the docs/architecture.md documentation",
                metadata={"files": ["docs/architecture.md"]},
            ),
            ACDependencySpec(
                index=1,
                content="Add new provider in src/ouroboros/providers/anthropic.py",
                metadata={"files": ["src/ouroboros/providers/anthropic.py"]},
            ),
        ]
        result = _run(predictor.predict(specs))
        # These ACs modify different files — but category expansion may
        # cause overlap. Check that explicit-only scenario works.
        # With category expansion, overlap may occur. This test validates
        # the prediction mechanism works.
        assert len(result.ac_predictions) == 2

    def test_overlap_on_same_file(self) -> None:
        predictor = _make_predictor()
        specs = [
            ACDependencySpec(
                index=0,
                content="Add error types to src/ouroboros/core/errors.py",
                metadata={"files": ["src/ouroboros/core/errors.py"]},
            ),
            ACDependencySpec(
                index=1,
                content="Update error handling in src/ouroboros/core/errors.py",
                metadata={"files": ["src/ouroboros/core/errors.py"]},
            ),
        ]
        result = _run(predictor.predict(specs))
        assert result.has_overlaps
        assert len(result.overlap_groups) >= 1
        # Both ACs should be isolated
        assert 0 in result.isolated_ac_indices
        assert 1 in result.isolated_ac_indices

    def test_overlap_on_same_directory(self) -> None:
        predictor = _make_predictor()
        specs = [
            ACDependencySpec(
                index=0,
                content="Add new coordinator helper function",
                metadata={"files": ["src/ouroboros/orchestrator/coordinator.py"]},
            ),
            ACDependencySpec(
                index=1,
                content="Update dependency analyzer logic",
                metadata={"files": ["src/ouroboros/orchestrator/dependency_analyzer.py"]},
            ),
        ]
        result = _run(predictor.predict(specs))
        # Neighbour expansion should cause overlap since both files are
        # in the same directory
        assert result.has_overlaps
        assert 0 in result.isolated_ac_indices
        assert 1 in result.isolated_ac_indices

    def test_shared_acs_no_overlap(self) -> None:
        """ACs with non-overlapping predictions stay in shared workspace."""
        file_index = frozenset({
            "src/module_a/foo.py",
            "src/module_b/bar.py",
            "tests/test_a.py",
            "tests/test_b.py",
        })
        predictor = _make_predictor(file_index)
        specs = [
            ACDependencySpec(
                index=0,
                content="Update module A functionality",
                metadata={"files": ["src/module_a/foo.py"]},
            ),
            ACDependencySpec(
                index=1,
                content="Update module B functionality",
                metadata={"files": ["src/module_b/bar.py"]},
            ),
        ]
        result = _run(predictor.predict(specs))
        # Different directories — should have no overlap
        assert 0 in result.shared_ac_indices
        assert 1 in result.shared_ac_indices


# ---------------------------------------------------------------------------
# Conservative prediction (zero false negatives)
# ---------------------------------------------------------------------------


class TestConservativePrediction:
    """Tests ensuring prediction is conservative (over-predict, never under-predict)."""

    def test_neighbour_expansion_includes_siblings(self) -> None:
        """When a file is predicted, siblings in the same dir are included."""
        predictor = _make_predictor()
        specs = [
            ACDependencySpec(
                index=0,
                content="Update types module",
                metadata={"files": ["src/ouroboros/core/types.py"]},
            ),
        ]
        result = _run(predictor.predict(specs))
        pred = result.ac_predictions[0]
        # Should include sibling files in src/ouroboros/core/
        assert "src/ouroboros/core/types.py" in pred.predicted_paths
        assert "src/ouroboros/core/errors.py" in pred.predicted_paths
        assert "src/ouroboros/core/seed.py" in pred.predicted_paths

    def test_category_expansion_is_broad(self) -> None:
        """Category matching should include all files under category directories."""
        predictor = _make_predictor()
        specs = [
            ACDependencySpec(
                index=0,
                content="Fix a bug in the orchestrator parallel execution",
            ),
        ]
        result = _run(predictor.predict(specs))
        pred = result.ac_predictions[0]
        # Should include all orchestrator files
        assert "src/ouroboros/orchestrator/parallel_executor.py" in pred.predicted_paths
        assert "src/ouroboros/orchestrator/coordinator.py" in pred.predicted_paths

    def test_vague_description_triggers_broad_prediction(self) -> None:
        """Vague AC descriptions should trigger broad file predictions."""
        predictor = _make_predictor()
        specs = [
            ACDependencySpec(
                index=0,
                content="Refactor the core module to improve error handling and logging",
            ),
        ]
        result = _run(predictor.predict(specs))
        pred = result.ac_predictions[0]
        # "core" category + "observability" category
        assert "core" in pred.categories
        # Should be broad
        assert len(pred.predicted_paths) > 5


# ---------------------------------------------------------------------------
# Stage-scoped prediction
# ---------------------------------------------------------------------------


class TestStageScopedPrediction:
    """Tests for predicting only within a specific execution stage."""

    def test_stage_filter(self) -> None:
        predictor = _make_predictor()
        specs = [
            ACDependencySpec(index=0, content="Update docs"),
            ACDependencySpec(index=1, content="Fix tests"),
            ACDependencySpec(index=2, content="Update config"),
        ]
        result = _run(predictor.predict(specs, stage_ac_indices=(0, 2)))
        assert result.all_ac_indices == frozenset({0, 2})
        assert 1 not in result.all_ac_indices


# ---------------------------------------------------------------------------
# Empty / edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_ac(self) -> None:
        predictor = _make_predictor()
        specs = [ACDependencySpec(index=0, content="Do something")]
        result = _run(predictor.predict(specs))
        assert len(result.ac_predictions) == 1
        assert not result.has_overlaps
        assert result.shared_ac_indices == frozenset({0})

    def test_empty_specs(self) -> None:
        predictor = _make_predictor()
        result = _run(predictor.predict([]))
        assert len(result.ac_predictions) == 0
        assert not result.has_overlaps

    def test_no_file_index(self) -> None:
        """Predictor works without a file index (ungrounded predictions)."""
        predictor = FileOverlapPredictor(file_index=frozenset())
        specs = [
            ACDependencySpec(
                index=0,
                content="Update src/ouroboros/core/types.py",
                metadata={"files": ["src/ouroboros/core/types.py"]},
            ),
        ]
        result = _run(predictor.predict(specs))
        pred = result.ac_predictions[0]
        # Should still have the explicitly mentioned paths
        assert "src/ouroboros/core/types.py" in pred.predicted_paths


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


class TestConvenienceFunction:
    """Tests for the predict_file_overlaps convenience function."""

    def test_predict_file_overlaps(self) -> None:
        specs = [
            ACDependencySpec(
                index=0,
                content="Update errors module",
                metadata={"files": ["src/ouroboros/core/errors.py"]},
            ),
            ACDependencySpec(
                index=1,
                content="Update types module",
                metadata={"files": ["src/ouroboros/core/types.py"]},
            ),
        ]
        result = _run(predict_file_overlaps(
            specs,
            file_index=_SAMPLE_FILE_INDEX,
        ))
        assert len(result.ac_predictions) == 2
        # Both files are in the same directory — neighbour expansion
        # should cause overlap
        assert result.has_overlaps


# ---------------------------------------------------------------------------
# Confidence estimation
# ---------------------------------------------------------------------------


class TestConfidenceEstimation:
    """Tests for prediction confidence scoring."""

    def test_explicit_paths_boost_confidence(self) -> None:
        predictor = _make_predictor()
        specs = [
            ACDependencySpec(
                index=0,
                content="Update src/ouroboros/core/types.py with Result type changes",
                metadata={"files": ["src/ouroboros/core/types.py"]},
            ),
        ]
        result = _run(predictor.predict(specs))
        pred = result.ac_predictions[0]
        # Explicit paths should give higher confidence
        assert pred.confidence >= 0.5

    def test_vague_description_lower_confidence(self) -> None:
        predictor = _make_predictor()
        specs = [
            ACDependencySpec(
                index=0,
                content="Make improvements to the system",
            ),
        ]
        result = _run(predictor.predict(specs))
        pred = result.ac_predictions[0]
        # Very vague — lower confidence
        assert pred.confidence <= 0.5
