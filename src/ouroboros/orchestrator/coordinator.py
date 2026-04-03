"""Level Coordinator for inter-level review and conflict resolution.

Detects file conflicts from parallel AC execution results and optionally
invokes a Claude session to auto-resolve them. Acts as an intelligent
review gate between dependency levels.

Architecture: Approach A (Pragmatic)
- Pure Python conflict detection from in-memory ACExecutionResult data
- Claude session only when file conflicts are detected
- Zero cost when no conflicts exist

Usage:
    coordinator = LevelCoordinator(adapter)
    conflicts = coordinator.detect_file_conflicts(level_results)

    if conflicts:
        review = await coordinator.run_review(
            execution_id="exec_123",
            conflicts=conflicts,
            level_context=level_ctx,
            level_number=1,
        )
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
import json
import re
from typing import TYPE_CHECKING, Any

from ouroboros.observability.logging import get_logger
from ouroboros.orchestrator.adapter import RuntimeHandle
from ouroboros.orchestrator.execution_runtime_scope import (
    build_level_coordinator_runtime_scope,
)

if TYPE_CHECKING:
    from ouroboros.orchestrator.adapter import AgentMessage, AgentRuntime
    from ouroboros.orchestrator.file_overlap_predictor import FileOverlapPrediction
    from ouroboros.orchestrator.level_context import LevelContext
    from ouroboros.orchestrator.parallel_executor import ACExecutionResult

log = get_logger(__name__)

_LEVEL_COORDINATOR_SESSION_KIND = "level_coordinator"
_COORDINATOR_SCOPE = "level"
_COORDINATOR_SESSION_ROLE = "coordinator"
_COORDINATOR_ARTIFACT_TYPE = "coordinator_review"


# Tools available to the Coordinator Claude session
COORDINATOR_TOOLS: list[str] = ["Read", "Bash", "Edit", "Grep", "Glob"]

# System prompt for the Coordinator agent
COORDINATOR_SYSTEM_PROMPT = (
    "You are a Level Coordinator reviewing parallel AC execution results. "
    "Your job is to detect and resolve file conflicts, then provide actionable "
    "guidance for the next level of execution. Be concise and precise."
)


@dataclass(frozen=True, slots=True)
class FileConflict:
    """A file modified by multiple ACs in the same level.

    Attributes:
        file_path: Path to the conflicting file.
        ac_indices: Which ACs modified this file.
        resolved: Whether the conflict was resolved by the Coordinator.
        resolution_description: How the conflict was resolved.
    """

    file_path: str
    ac_indices: tuple[int, ...]
    resolved: bool = False
    resolution_description: str = ""


@dataclass(frozen=True, slots=True)
class CoordinatorReview:
    """Result of a Coordinator review between dependency levels.

    Attributes:
        level_number: Which level was reviewed.
        conflicts_detected: File conflicts found.
        review_summary: Coordinator's analysis text.
        fixes_applied: Descriptions of fixes made.
        warnings_for_next_level: Injected into next level prompt.
        duration_seconds: Time spent on review.
        session_id: Claude session ID (None if no session was needed).
        session_scope_id: Stable identity for persisted reconciliation runtime state.
        session_state_path: Stable state path for persisted reconciliation runtime state.
        final_output: Raw final coordinator output captured for level-scoped artifacts.
        messages: Runtime messages retained in memory for normalized audit emission.
    """

    level_number: int
    conflicts_detected: tuple[FileConflict, ...] = field(default_factory=tuple)
    review_summary: str = ""
    fixes_applied: tuple[str, ...] = field(default_factory=tuple)
    warnings_for_next_level: tuple[str, ...] = field(default_factory=tuple)
    duration_seconds: float = 0.0
    session_id: str | None = None
    session_scope_id: str | None = None
    session_state_path: str | None = None
    final_output: str = ""
    messages: tuple[AgentMessage, ...] = field(default_factory=tuple)

    @property
    def scope(self) -> str:
        """Coordinator reconciliation is always attributed at level scope."""
        return _COORDINATOR_SCOPE

    @property
    def session_role(self) -> str:
        """Coordinator reconciliation never impersonates an AC session."""
        return _COORDINATOR_SESSION_ROLE

    @property
    def stage_index(self) -> int:
        """Return the 0-based execution stage index for this level."""
        return self.level_number - 1

    @property
    def artifact_type(self) -> str:
        """Return the persisted artifact type for coordinator output."""
        return _COORDINATOR_ARTIFACT_TYPE

    @property
    def artifact_owner(self) -> str:
        """Coordinator artifacts are owned by the level coordinator."""
        return _COORDINATOR_SESSION_ROLE

    @property
    def artifact_scope(self) -> str:
        """Coordinator artifacts belong to the shared level workspace state."""
        return _COORDINATOR_SCOPE

    @property
    def artifact_owner_id(self) -> str:
        """Return the stable coordinator scope identifier used for persistence."""
        if self.session_scope_id:
            return self.session_scope_id
        return f"level_{self.level_number}_coordinator_reconciliation"

    @property
    def artifact_state_path(self) -> str:
        """Return the stable persistence path for coordinator runtime state."""
        if self.session_state_path:
            return self.session_state_path
        return f"execution.levels.level_{self.level_number}.coordinator_reconciliation_session"

    def to_artifact_payload(self) -> dict[str, Any]:
        """Build normalized persisted artifact metadata for coordinator output."""
        return {
            "scope": self.scope,
            "session_role": self.session_role,
            "stage_index": self.stage_index,
            "level_number": self.level_number,
            "session_scope_id": self.artifact_owner_id,
            "session_state_path": self.artifact_state_path,
            "artifact_scope": self.artifact_scope,
            "artifact_owner": self.artifact_owner,
            "artifact_owner_id": self.artifact_owner_id,
            "artifact": self.final_output,
            "artifact_type": self.artifact_type,
        }


@dataclass(frozen=True, slots=True)
class ConflictPredictionMetrics:
    """Tracks prediction accuracy for a single level's conflict detection.

    Compares the prediction engine's pre-execution file overlap predictions
    against the coordinator's post-hoc conflict detection to measure
    prediction quality. Conservative prediction means ``missed_conflicts``
    should always be zero.

    Attributes:
        level_number: Which level these metrics apply to.
        predicted_overlap_files: Files that the prediction engine flagged
            as potential overlaps before execution.
        actual_conflict_files: Files that the coordinator detected as
            actual conflicts post-execution.
        covered_conflicts: Actual conflicts that were covered by prediction
            (intersection of predicted and actual).
        missed_conflicts: Actual conflicts NOT covered by prediction —
            indicates a false negative in the prediction engine.
        false_positive_files: Predicted overlaps that did not become actual
            conflicts — expected in conservative prediction.
        prediction_was_active: Whether prediction was run for this level.
        isolation_was_applied: Whether worktree isolation was actually applied
            for any AC in this level.
    """

    level_number: int = 0
    predicted_overlap_files: frozenset[str] = field(default_factory=frozenset)
    actual_conflict_files: frozenset[str] = field(default_factory=frozenset)
    covered_conflicts: frozenset[str] = field(default_factory=frozenset)
    missed_conflicts: frozenset[str] = field(default_factory=frozenset)
    false_positive_files: frozenset[str] = field(default_factory=frozenset)
    prediction_was_active: bool = False
    isolation_was_applied: bool = False

    @property
    def prediction_covered_all(self) -> bool:
        """True when prediction caught every actual conflict (zero misses)."""
        return len(self.missed_conflicts) == 0

    @property
    def safety_net_fired(self) -> bool:
        """True when post-hoc coordinator detected conflicts not covered by prediction."""
        return len(self.missed_conflicts) > 0

    def to_metadata(self) -> dict[str, Any]:
        """Serialize for event/checkpoint storage."""
        return {
            "level_number": self.level_number,
            "predicted_overlap_count": len(self.predicted_overlap_files),
            "actual_conflict_count": len(self.actual_conflict_files),
            "covered_count": len(self.covered_conflicts),
            "missed_count": len(self.missed_conflicts),
            "false_positive_count": len(self.false_positive_files),
            "prediction_was_active": self.prediction_was_active,
            "isolation_was_applied": self.isolation_was_applied,
            "prediction_covered_all": self.prediction_covered_all,
            "safety_net_fired": self.safety_net_fired,
            "missed_files": sorted(self.missed_conflicts) if self.missed_conflicts else [],
        }


class LevelCoordinator:
    """Coordinates between parallel execution levels.

    Detects file conflicts from AC execution results and optionally
    invokes Claude to resolve them.
    """

    def __init__(
        self,
        adapter: AgentRuntime,
        inherited_runtime_handle: RuntimeHandle | None = None,
        task_cwd: str | None = None,
    ) -> None:
        """Initialize coordinator.

        Args:
            adapter: Agent runtime for conflict resolution sessions.
            inherited_runtime_handle: Optional parent Claude runtime handle for
                        delegated child executions.
        """
        self._adapter = adapter
        self._inherited_runtime_handle = inherited_runtime_handle
        self._task_cwd = task_cwd
        self._level_runtime_handles: dict[tuple[str, int], RuntimeHandle] = {}

    def _build_level_runtime_handle(
        self,
        execution_id: str,
        level_number: int,
        *,
        previous_review: CoordinatorReview | None = None,
    ) -> RuntimeHandle | None:
        """Build or resume the runtime handle for level-scoped coordinator work."""
        runtime_scope = build_level_coordinator_runtime_scope(execution_id, level_number)
        cache_key = (execution_id, level_number)
        seeded_handle = self._level_runtime_handles.get(cache_key)
        backend = self._adapter.runtime_backend
        if not backend:
            # Fallback: use inherited runtime handle if available
            return self._inherited_runtime_handle

        cwd = self._task_cwd or self._adapter.working_directory
        approval_mode = self._adapter.permission_mode
        native_session_id = seeded_handle.native_session_id if seeded_handle is not None else None
        if native_session_id is None and previous_review is not None:
            if previous_review.level_number == level_number:
                native_session_id = previous_review.session_id

        metadata: dict[str, object] = (
            dict(seeded_handle.metadata) if seeded_handle is not None else {}
        )
        metadata.update(
            {
                "scope": "level",
                "execution_id": execution_id,
                "level_number": level_number,
                "session_role": "coordinator",
                "session_scope_id": runtime_scope.aggregate_id,
                "session_state_path": runtime_scope.state_path,
            }
        )
        if seeded_handle is not None:
            return replace(
                seeded_handle,
                backend=backend,
                kind=seeded_handle.kind or _LEVEL_COORDINATOR_SESSION_KIND,
                native_session_id=native_session_id,
                cwd=(
                    seeded_handle.cwd
                    if seeded_handle.cwd
                    else cwd
                    if isinstance(cwd, str) and cwd
                    else None
                ),
                approval_mode=(
                    seeded_handle.approval_mode
                    if seeded_handle.approval_mode
                    else approval_mode
                    if isinstance(approval_mode, str) and approval_mode
                    else None
                ),
                updated_at=datetime.now(UTC).isoformat(),
                metadata=metadata,
            )

        return RuntimeHandle(
            backend=backend,
            kind=_LEVEL_COORDINATOR_SESSION_KIND,
            native_session_id=native_session_id,
            cwd=cwd if isinstance(cwd, str) and cwd else None,
            approval_mode=approval_mode
            if isinstance(approval_mode, str) and approval_mode
            else None,
            updated_at=datetime.now(UTC).isoformat(),
            metadata=metadata,
        )

    def _remember_level_runtime_handle(
        self,
        execution_id: str,
        level_number: int,
        runtime_handle: RuntimeHandle | None,
    ) -> None:
        """Cache the latest runtime handle for repeated same-level reconciliation."""
        if runtime_handle is None:
            return
        self._level_runtime_handles[(execution_id, level_number)] = runtime_handle

    async def run_pre_execution_prediction(
        self,
        ac_specs: Any,
        *,
        seed: Any | None = None,
        stage_ac_indices: tuple[int, ...] | None = None,
        repo_root: str | None = None,
    ) -> FileOverlapPrediction | None:
        """Run the file overlap prediction engine before AC execution.

        This is the pre-execution phase: predict which files each AC will
        touch and identify overlap groups. The prediction result drives
        downstream isolation decisions (worktree vs shared workspace).

        Returns None if prediction cannot be run (e.g. no AC specs).

        Args:
            ac_specs: AC dependency specs for the current stage.
            seed: Optional seed for brownfield context integration.
            stage_ac_indices: AC indices in this stage.
            repo_root: Repository root for filesystem discovery.

        Returns:
            FileOverlapPrediction or None if prediction was skipped.
        """
        from ouroboros.orchestrator.file_overlap_predictor import predict_file_overlaps

        if not ac_specs:
            return None

        try:
            prediction = await predict_file_overlaps(
                ac_specs,
                repo_root=repo_root,
                seed=seed,
                stage_ac_indices=stage_ac_indices,
            )
            log.info(
                "coordinator.prediction.completed",
                has_overlaps=prediction.has_overlaps,
                overlap_groups=len(prediction.overlap_groups),
                isolated_count=len(prediction.isolated_ac_indices),
                shared_count=len(prediction.shared_ac_indices),
            )
            return prediction
        except Exception as exc:
            log.warning(
                "coordinator.prediction.failed",
                error=str(exc),
            )
            return None

    @staticmethod
    def build_prediction_metrics(
        level_number: int,
        prediction: FileOverlapPrediction | None,
        conflicts: list[FileConflict],
        *,
        isolation_was_applied: bool = False,
    ) -> ConflictPredictionMetrics:
        """Compare prediction results against actual post-hoc conflicts.

        Builds metrics that track whether the prediction engine covered
        all actual file conflicts, or whether the coordinator safety net
        had to catch conflicts that prediction missed.

        Args:
            level_number: Which level these metrics apply to.
            prediction: Pre-execution prediction result (None if skipped).
            conflicts: Post-hoc detected file conflicts.
            isolation_was_applied: Whether worktree isolation was used.

        Returns:
            ConflictPredictionMetrics with coverage analysis.
        """
        actual_conflict_files = frozenset(c.file_path for c in conflicts)

        if prediction is None:
            # No prediction was run — all conflicts are "missed" by definition
            return ConflictPredictionMetrics(
                level_number=level_number,
                actual_conflict_files=actual_conflict_files,
                missed_conflicts=actual_conflict_files,
                prediction_was_active=False,
                isolation_was_applied=isolation_was_applied,
            )

        # Collect all files that prediction flagged as overlapping
        predicted_files: set[str] = set()
        for group in prediction.overlap_groups:
            predicted_files.update(group.shared_paths)

        predicted_overlap_files = frozenset(predicted_files)
        covered = actual_conflict_files & predicted_overlap_files
        missed = actual_conflict_files - predicted_overlap_files
        false_positives = predicted_overlap_files - actual_conflict_files

        metrics = ConflictPredictionMetrics(
            level_number=level_number,
            predicted_overlap_files=predicted_overlap_files,
            actual_conflict_files=actual_conflict_files,
            covered_conflicts=covered,
            missed_conflicts=missed,
            false_positive_files=false_positives,
            prediction_was_active=True,
            isolation_was_applied=isolation_was_applied,
        )

        # Log prediction accuracy
        if missed:
            log.warning(
                "coordinator.prediction_metrics.missed_conflicts",
                level=level_number,
                missed_count=len(missed),
                missed_files=sorted(missed),
                covered_count=len(covered),
                predicted_count=len(predicted_overlap_files),
                actual_count=len(actual_conflict_files),
            )
        elif actual_conflict_files:
            log.info(
                "coordinator.prediction_metrics.all_covered",
                level=level_number,
                covered_count=len(covered),
                false_positive_count=len(false_positives),
                isolation_was_applied=isolation_was_applied,
            )
        else:
            log.debug(
                "coordinator.prediction_metrics.no_conflicts",
                level=level_number,
                predicted_overlap_count=len(predicted_overlap_files),
                false_positive_count=len(false_positives),
            )

        return metrics

    @staticmethod
    def detect_file_conflicts(
        level_results: list[ACExecutionResult],
    ) -> list[FileConflict]:
        """Detect files modified by multiple ACs in the same level.

        Scans ACExecutionResult.messages for Write/Edit tool calls and
        identifies files touched by more than one AC.

        Args:
            level_results: Results from ACs executed in the same level.

        Returns:
            List of FileConflict for files modified by 2+ ACs.
        """
        # Map file_path → set of ac_indices that modified it
        file_to_acs: dict[str, set[int]] = defaultdict(set)

        for result in level_results:
            _collect_file_modifications(result, file_to_acs)

        # Filter to files with 2+ writers
        conflicts: list[FileConflict] = []
        for file_path, ac_indices in sorted(file_to_acs.items()):
            if len(ac_indices) >= 2:
                conflicts.append(
                    FileConflict(
                        file_path=file_path,
                        ac_indices=tuple(sorted(ac_indices)),
                    )
                )

        if conflicts:
            log.warning(
                "coordinator.conflicts_detected",
                conflict_count=len(conflicts),
                files=[c.file_path for c in conflicts],
            )
        else:
            log.info("coordinator.no_conflicts")

        return conflicts

    async def run_review(
        self,
        execution_id: str,
        conflicts: list[FileConflict],
        level_context: LevelContext,
        level_number: int,
        *,
        previous_review: CoordinatorReview | None = None,
    ) -> CoordinatorReview:
        """Run a Claude session to review and resolve file conflicts.

        Only called when conflicts are detected (Approach A).

        Args:
            conflicts: Detected file conflicts.
            level_context: Context from the completed level.
            level_number: Which level was just completed.

        Returns:
            CoordinatorReview with resolution details.
        """
        start_time = datetime.now(UTC)
        runtime_scope = build_level_coordinator_runtime_scope(execution_id, level_number)

        prompt = _build_review_prompt(conflicts, level_context, level_number)

        log.info(
            "coordinator.review.started",
            level=level_number,
            conflict_count=len(conflicts),
        )

        runtime_handle = self._build_level_runtime_handle(
            execution_id,
            level_number,
            previous_review=previous_review,
        )
        session_id: str | None = None
        final_text = ""
        messages: list[AgentMessage] = []

        try:
            async for message in self._adapter.execute_task(
                prompt=prompt,
                tools=COORDINATOR_TOOLS,
                system_prompt=COORDINATOR_SYSTEM_PROMPT,
                resume_handle=runtime_handle,
            ):
                messages.append(message)
                if message.resume_handle is not None:
                    runtime_handle = message.resume_handle
                    self._remember_level_runtime_handle(
                        execution_id,
                        level_number,
                        runtime_handle,
                    )
                if message.resume_handle is not None and message.resume_handle.native_session_id:
                    session_id = message.resume_handle.native_session_id
                elif message.data.get("session_id"):
                    session_id = message.data["session_id"]
                if message.is_final:
                    final_text = message.content
            self._remember_level_runtime_handle(execution_id, level_number, runtime_handle)

        except Exception as e:
            log.exception(
                "coordinator.review.failed",
                level=level_number,
                error=str(e),
            )
            self._remember_level_runtime_handle(execution_id, level_number, runtime_handle)
            duration = (datetime.now(UTC) - start_time).total_seconds()
            return CoordinatorReview(
                level_number=level_number,
                conflicts_detected=tuple(conflicts),
                review_summary=f"Coordinator review failed: {e}",
                duration_seconds=duration,
                session_scope_id=runtime_scope.aggregate_id,
                session_state_path=runtime_scope.state_path,
                session_id=session_id,
                final_output=f"Coordinator review failed: {e}",
                messages=tuple(messages),
            )

        duration = (datetime.now(UTC) - start_time).total_seconds()

        # Parse structured response from Claude
        review = replace(
            _parse_review_response(
                final_text,
                conflicts,
                level_number,
                duration,
                session_id,
                session_scope_id=runtime_scope.aggregate_id,
                session_state_path=runtime_scope.state_path,
            ),
            final_output=final_text,
            messages=tuple(messages),
        )

        log.info(
            "coordinator.review.completed",
            level=level_number,
            fixes_applied=len(review.fixes_applied),
            warnings=len(review.warnings_for_next_level),
            duration_seconds=duration,
        )

        return review


def _collect_file_modifications(
    result: ACExecutionResult,
    file_to_acs: dict[str, set[int]],
) -> None:
    """Recursively collect file modifications from an AC result.

    Handles both atomic and decomposed (Sub-AC) results.

    Args:
        result: AC execution result to scan.
        file_to_acs: Accumulator mapping file_path → ac_indices.
    """
    # Check direct messages for Write/Edit tool calls
    for msg in result.messages:
        if msg.tool_name in ("Write", "Edit"):
            tool_input = msg.data.get("tool_input", {})
            file_path = tool_input.get("file_path")
            if file_path:
                file_to_acs.setdefault(file_path, set()).add(result.ac_index)

    # Recurse into Sub-AC results
    for sub_result in result.sub_results:
        # Sub-ACs inherit the parent AC index for conflict tracking
        for msg in sub_result.messages:
            if msg.tool_name in ("Write", "Edit"):
                tool_input = msg.data.get("tool_input", {})
                file_path = tool_input.get("file_path")
                if file_path:
                    file_to_acs.setdefault(file_path, set()).add(result.ac_index)


def _build_review_prompt(
    conflicts: list[FileConflict],
    level_context: LevelContext,
    level_number: int,
) -> str:
    """Build the prompt for the Coordinator Claude session.

    Args:
        conflicts: Detected file conflicts.
        level_context: Context from the completed level.
        level_number: Which level was just completed.

    Returns:
        Formatted prompt string.
    """
    context_text = level_context.to_prompt_text()

    conflict_lines: list[str] = []
    for conflict in conflicts:
        ac_list = ", ".join(f"AC {i + 1}" for i in conflict.ac_indices)
        conflict_lines.append(f"- `{conflict.file_path}` modified by: {ac_list}")
    conflict_text = "\n".join(conflict_lines)

    return f"""Review the results of Level {level_number} parallel AC execution.

## Level {level_number} Results
{context_text}

## File Conflicts Detected
{conflict_text}

## Your Tasks
1. Read the conflicting files using the Read tool
2. Run `git diff` if needed to understand changes
3. If edits from different ACs conflict, resolve them using the Edit tool
4. Provide your review as a structured JSON response:

```json
{{
  "review_summary": "Brief analysis of the level results",
  "fixes_applied": ["Description of fix 1", "..."],
  "warnings_for_next_level": ["Warning 1 for next ACs", "..."],
  "conflicts_resolved": ["{conflicts[0].file_path if conflicts else ""}"]
}}
```

Respond with the JSON block after completing your review and any fixes.
"""


def _parse_review_response(
    response_text: str,
    original_conflicts: list[FileConflict],
    level_number: int,
    duration: float,
    session_id: str | None,
    *,
    session_scope_id: str | None = None,
    session_state_path: str | None = None,
) -> CoordinatorReview:
    """Parse the Coordinator's structured JSON response.

    Falls back to using the raw response as review_summary if JSON parsing fails.

    Args:
        response_text: Raw response from the Coordinator Claude session.
        original_conflicts: Original conflict list for resolution tracking.
        level_number: Level that was reviewed.
        duration: Time spent on review.
        session_id: Claude session ID.

    Returns:
        CoordinatorReview populated from the parsed response.
    """
    review_summary = ""
    fixes_applied: list[str] = []
    warnings: list[str] = []
    resolved_files: set[str] = set()

    # Try to extract JSON from the response
    json_match = re.search(r"```json\s*\n(.*?)\n```", response_text, re.DOTALL)
    if not json_match:
        # Try bare JSON object
        json_match = re.search(r"\{[^{}]*\}", response_text, re.DOTALL)

    if json_match:
        try:
            data = json.loads(
                json_match.group(1) if "```" in json_match.group() else json_match.group()
            )
            review_summary = data.get("review_summary", "")
            fixes_applied = data.get("fixes_applied", [])
            warnings = data.get("warnings_for_next_level", [])
            resolved_files = set(data.get("conflicts_resolved", []))
        except (json.JSONDecodeError, IndexError):
            log.warning(
                "coordinator.parse_failed",
                response_preview=response_text[:200],
            )

    # Fallback: use raw text as summary
    if not review_summary:
        review_summary = response_text[:500].strip() if response_text else "No review output"

    # Mark conflicts as resolved based on Coordinator's report
    updated_conflicts: list[FileConflict] = []
    for conflict in original_conflicts:
        is_resolved = conflict.file_path in resolved_files
        updated_conflicts.append(
            FileConflict(
                file_path=conflict.file_path,
                ac_indices=conflict.ac_indices,
                resolved=is_resolved,
                resolution_description="Resolved by Coordinator" if is_resolved else "",
            )
        )

    return CoordinatorReview(
        level_number=level_number,
        conflicts_detected=tuple(updated_conflicts),
        review_summary=review_summary,
        fixes_applied=tuple(fixes_applied),
        warnings_for_next_level=tuple(warnings),
        duration_seconds=duration,
        session_id=session_id,
        session_scope_id=session_scope_id,
        session_state_path=session_state_path,
    )


__all__ = [
    "COORDINATOR_TOOLS",
    "ConflictPredictionMetrics",
    "CoordinatorReview",
    "FileConflict",
    "LevelCoordinator",
]
