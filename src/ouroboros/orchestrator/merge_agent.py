"""Merge-agent invocation for resolving git merge conflicts from parallel AC isolation.

When git auto-merge fails (conflicting hunks in the same file regions), this
module invokes a Claude agent session to resolve the conflicts. The agent
receives the conflict context (files, diff, branch info) and uses Read/Edit
tools to produce a clean resolution.

Architecture:
- WorktreeMerger (worktree_merge.py) detects conflicts and returns MergeResult
- This module takes CONFLICT MergeResults and dispatches a merge-agent
- The merge-agent operates in the repo working tree with conflict markers present
- After resolution, changes are committed and the merge is completed
- The existing LevelCoordinator remains as a safety-net fallback layer

The merge-agent flags warnings for non-trivial resolutions so that the
downstream 3-stage evaluation pipeline can verify correctness.

Usage:
    agent = MergeAgentInvoker(adapter=adapter, repo_root="/path/to/repo")

    resolution = await agent.resolve_conflicts(
        merge_result=conflict_result,
        target_branch="ooo/orch_abc123",
        execution_id="orch_abc123",
        level_number=1,
    )

    if resolution.resolved:
        print(f"Resolved {len(resolution.files_resolved)} files")
    else:
        print(f"Failed: {resolution.error_message}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ouroboros.core.worktree import WorktreeError, _run_git, _run_git_process
from ouroboros.observability.logging import get_logger
from ouroboros.orchestrator.execution_runtime_scope import ExecutionRuntimeScope
from ouroboros.orchestrator.worktree_merge import MergeOutcome, MergeResult

if TYPE_CHECKING:
    from ouroboros.orchestrator.adapter import AgentMessage, AgentRuntime, RuntimeHandle

log = get_logger(__name__)

# Tools available to the merge-agent Claude session
MERGE_AGENT_TOOLS: list[str] = ["Read", "Edit", "Bash", "Grep", "Glob"]

_MERGE_AGENT_SESSION_KIND = "merge_agent"
_MERGE_AGENT_SCOPE = "merge"
_MERGE_AGENT_SESSION_ROLE = "merge_agent"


class MergeResolutionOutcome(Enum):
    """Outcome of a merge-agent resolution attempt."""

    RESOLVED = "resolved"
    """Agent successfully resolved all conflicts."""

    PARTIAL = "partial"
    """Agent resolved some but not all conflicts."""

    FAILED = "failed"
    """Agent could not resolve the conflicts."""

    SKIPPED = "skipped"
    """No conflicts to resolve (input was not a CONFLICT result)."""


@dataclass(frozen=True, slots=True)
class MergeResolution:
    """Result of a merge-agent conflict resolution attempt.

    Attributes:
        ac_index: 0-based AC index whose merge had conflicts.
        ac_branch: Git branch that was being merged.
        outcome: Whether the resolution succeeded, partially succeeded, or failed.
        files_resolved: Files that were successfully resolved.
        files_remaining: Files that still have unresolved conflicts.
        merge_sha: Commit SHA after successful merge completion (None otherwise).
        warnings: Non-trivial resolution warnings for downstream verification.
        agent_summary: Agent's description of what it resolved and how.
        error_message: Error details when outcome is FAILED.
        duration_seconds: Time spent on resolution.
        session_id: Claude session ID for the merge-agent.
        messages: Runtime messages from the merge-agent session.
    """

    ac_index: int
    ac_branch: str
    outcome: MergeResolutionOutcome
    files_resolved: tuple[str, ...] = field(default_factory=tuple)
    files_remaining: tuple[str, ...] = field(default_factory=tuple)
    merge_sha: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)
    agent_summary: str = ""
    error_message: str = ""
    duration_seconds: float = 0.0
    session_id: str | None = None
    messages: tuple[AgentMessage, ...] = field(default_factory=tuple)

    @property
    def resolved(self) -> bool:
        """True when all conflicts were resolved."""
        return self.outcome == MergeResolutionOutcome.RESOLVED

    @property
    def has_warnings(self) -> bool:
        """True when the resolution includes non-trivial warnings."""
        return len(self.warnings) > 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for event/checkpoint storage."""
        return {
            "ac_index": self.ac_index,
            "ac_branch": self.ac_branch,
            "outcome": self.outcome.value,
            "files_resolved": list(self.files_resolved),
            "files_remaining": list(self.files_remaining),
            "merge_sha": self.merge_sha,
            "warnings": list(self.warnings),
            "agent_summary": self.agent_summary,
            "error_message": self.error_message,
            "duration_seconds": self.duration_seconds,
        }


def _build_merge_agent_runtime_scope(
    execution_id: str,
    level_number: int,
    ac_index: int,
) -> ExecutionRuntimeScope:
    """Build the persisted runtime scope for merge-agent work."""
    return ExecutionRuntimeScope(
        aggregate_type="execution",
        aggregate_id=f"{execution_id}_level_{level_number}_merge_ac_{ac_index}",
        state_path=(
            f"execution.workflows.{execution_id}.levels.level_{level_number}."
            f"merge_agent_ac_{ac_index}"
        ),
    )


# System prompt for the merge-agent
MERGE_AGENT_SYSTEM_PROMPT = """\
You are a Merge Agent responsible for resolving git merge conflicts between \
parallel acceptance criteria (AC) branches.

Your task:
1. Read each conflicting file to understand both sides of the conflict.
2. Resolve the conflict by combining BOTH sides' intent — never discard \
either AC's work unless it is truly redundant.
3. Use the Edit tool to write the resolved content (removing all conflict \
markers like <<<<<<<, =======, >>>>>>>).
4. After resolving ALL files, verify no conflict markers remain by searching \
for "<<<<<<< " in the resolved files.

Rules:
- PRESERVE all meaningful changes from both sides of each conflict.
- When both sides add different code to the same location, include BOTH \
additions in a logical order.
- When both sides modify the same line differently, combine the intent of \
both modifications.
- If a conflict is genuinely incompatible (rare), choose the version that \
best serves the overall goal and add a WARNING comment explaining the choice.
- Do NOT run tests or compile — that belongs to the evaluation pipeline.
- Be concise in your explanations.

Output format (at the end of your work):
RESOLUTION_SUMMARY:
- For each file: what you resolved and how
- Any warnings about non-trivial choices made
RESOLUTION_STATUS: RESOLVED | PARTIAL | FAILED
"""


def _build_merge_prompt(
    merge_result: MergeResult,
    target_branch: str,
    conflict_file_contents: dict[str, str],
) -> str:
    """Build the prompt for the merge-agent with full conflict context.

    Args:
        merge_result: The CONFLICT MergeResult from WorktreeMerger.
        target_branch: The branch being merged into.
        conflict_file_contents: Pre-read contents of each conflicting file
            (with conflict markers present in the working tree).

    Returns:
        Formatted prompt string for the merge-agent.
    """
    files_section = "\n".join(
        f"  - {f}" for f in merge_result.conflicting_files
    )

    contents_section = ""
    for filepath, content in conflict_file_contents.items():
        # Truncate very large files to avoid prompt overflow
        truncated = content[:8000] if len(content) > 8000 else content
        suffix = "\n... (truncated)" if len(content) > 8000 else ""
        contents_section += f"\n### {filepath}\n```\n{truncated}{suffix}\n```\n"

    diff_section = merge_result.conflict_diff
    if len(diff_section) > 6000:
        diff_section = diff_section[:6000] + "\n... (diff truncated)"

    return f"""\
## Merge Conflict Resolution

**Target branch:** `{target_branch}`
**Source branch (AC {merge_result.ac_index}):** `{merge_result.ac_branch}`

### Conflicting files:
{files_section}

### Conflict diff from git:
```diff
{diff_section}
```

### Current file contents (with conflict markers):
{contents_section}

Please resolve ALL conflicts in the files listed above. Remove all conflict \
markers (<<<<<<<, =======, >>>>>>>) and produce clean, working code that \
preserves the intent of both branches.
"""


def _read_conflict_file_contents(
    repo_root: Path,
    conflicting_files: tuple[str, ...],
) -> dict[str, str]:
    """Read the contents of conflicting files (with conflict markers).

    Returns a dict mapping file path to contents. Files that cannot be read
    are silently skipped.
    """
    contents: dict[str, str] = {}
    for filepath in conflicting_files:
        full_path = repo_root / filepath
        try:
            contents[filepath] = full_path.read_text(errors="replace")
        except OSError:
            log.warning(
                "merge_agent.read_conflict_file_failed",
                filepath=filepath,
            )
    return contents


def _check_remaining_conflicts(
    repo_root: Path,
    conflicting_files: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    """Check which files still have conflict markers after resolution.

    Returns:
        (resolved_files, remaining_files) — files without/with conflict markers.
    """
    resolved: list[str] = []
    remaining: list[str] = []

    for filepath in conflicting_files:
        full_path = repo_root / filepath
        try:
            content = full_path.read_text(errors="replace")
        except OSError:
            remaining.append(filepath)
            continue

        if "<<<<<<< " in content or "=======" in content or ">>>>>>> " in content:
            remaining.append(filepath)
        else:
            resolved.append(filepath)

    return resolved, remaining


def _start_merge_with_conflicts(
    repo_root: Path,
    target_branch: str,
    ac_branch: str,
) -> bool:
    """Re-start the merge so conflict markers are present in the working tree.

    The WorktreeMerger aborts the merge after capturing conflict info. To let
    the merge-agent edit the conflicting files, we need to re-start the merge
    so git writes the conflict markers into the working tree.

    Returns True if the merge was started and conflicts are present.
    """
    # Ensure we're on the target branch
    current = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    if current != target_branch:
        _run_git(["checkout", target_branch], repo_root)

    # Start the merge (expected to fail with conflicts)
    result = _run_git_process(
        ["merge", "--no-edit", "--no-ff", ac_branch],
        repo_root,
    )

    # We expect this to fail with conflicts
    is_conflict = (
        "CONFLICT" in result.stdout
        or "CONFLICT" in result.stderr
        or "Automatic merge failed" in result.stdout
        or "Automatic merge failed" in result.stderr
    )

    if result.returncode == 0:
        # Unexpectedly succeeded — auto-merge worked this time
        log.info(
            "merge_agent.auto_merge_succeeded_on_retry",
            target_branch=target_branch,
            ac_branch=ac_branch,
        )
        return False

    if not is_conflict:
        # Non-conflict error — abort and bail
        _run_git_process(["merge", "--abort"], repo_root)
        log.error(
            "merge_agent.merge_start_error",
            target_branch=target_branch,
            ac_branch=ac_branch,
            stderr=result.stderr.strip(),
        )
        return False

    return True


def _complete_merge_after_resolution(
    repo_root: Path,
    ac_index: int,
    ac_branch: str,
) -> str | None:
    """Stage resolved files and complete the merge commit.

    Returns the merge commit SHA, or None if the commit failed.
    """
    try:
        _run_git(["add", "-A"], repo_root)
        _run_git(
            ["commit", "--no-edit", "-m", f"Merge AC {ac_index} ({ac_branch}) — agent-resolved"],
            repo_root,
        )
        sha = _run_git(["rev-parse", "HEAD"], repo_root)
        return sha
    except WorktreeError as exc:
        log.error(
            "merge_agent.commit_failed",
            ac_index=ac_index,
            ac_branch=ac_branch,
            error=str(exc),
        )
        return None


def _abort_merge(repo_root: Path) -> None:
    """Abort an in-progress merge to restore clean state."""
    _run_git_process(["merge", "--abort"], repo_root)


def _generate_resolution_warnings(
    agent_summary: str,
    files_resolved: list[str],
    files_remaining: list[str],
) -> list[str]:
    """Extract warnings from the agent's resolution for downstream verification.

    Non-trivial resolutions are flagged so the evaluation pipeline can verify.
    """
    warnings: list[str] = []

    if files_remaining:
        warnings.append(
            f"Merge-agent left {len(files_remaining)} unresolved file(s): "
            f"{', '.join(files_remaining)}"
        )

    # Flag non-trivial resolutions mentioned by the agent
    nontrivial_keywords = ["WARNING", "incompatible", "chose", "discarded", "dropped"]
    for keyword in nontrivial_keywords:
        if keyword.lower() in agent_summary.lower():
            warnings.append(
                f"Merge-agent flagged non-trivial resolution (keyword: {keyword}). "
                "Review recommended."
            )
            break

    if len(files_resolved) > 3:
        warnings.append(
            f"Merge-agent resolved {len(files_resolved)} files — "
            "large conflict scope increases risk."
        )

    return warnings


def _parse_agent_output(final_text: str) -> tuple[str, str]:
    """Parse the agent's final output for summary and status.

    Returns:
        (summary, status) where status is one of RESOLVED/PARTIAL/FAILED.
    """
    summary = ""
    status = "RESOLVED"

    # Extract RESOLUTION_SUMMARY
    if "RESOLUTION_SUMMARY:" in final_text:
        parts = final_text.split("RESOLUTION_SUMMARY:", 1)
        tail = parts[1]
        # Find where RESOLUTION_STATUS starts
        if "RESOLUTION_STATUS:" in tail:
            summary = tail.split("RESOLUTION_STATUS:", 1)[0].strip()
        else:
            summary = tail.strip()

    # Extract RESOLUTION_STATUS
    if "RESOLUTION_STATUS:" in final_text:
        status_line = final_text.split("RESOLUTION_STATUS:", 1)[1].strip()
        # Take first word/line
        status_word = status_line.split()[0].strip() if status_line.split() else "RESOLVED"
        if status_word.upper() in {"RESOLVED", "PARTIAL", "FAILED"}:
            status = status_word.upper()

    # If no structured output, use the full text as summary
    if not summary:
        summary = final_text[-500:] if len(final_text) > 500 else final_text

    return summary, status


class MergeAgentInvoker:
    """Invokes a Claude agent session to resolve git merge conflicts.

    When the WorktreeMerger returns a CONFLICT result, this class:
    1. Re-starts the merge so conflict markers are in the working tree
    2. Reads the conflicting file contents (with markers)
    3. Dispatches a Claude session with the conflict context
    4. Verifies the agent's resolution (no remaining conflict markers)
    5. Completes the merge commit if all conflicts are resolved
    6. Generates warnings for non-trivial resolutions

    Thread-safety: NOT thread-safe. Should be called sequentially from
    the orchestrator after parallel AC execution completes.
    """

    def __init__(
        self,
        adapter: AgentRuntime,
        repo_root: str | Path,
        *,
        runtime_handle: RuntimeHandle | None = None,
    ) -> None:
        """Initialize merge-agent invoker.

        Args:
            adapter: Agent runtime for Claude sessions.
            repo_root: Absolute path to the repository root.
            runtime_handle: Optional parent runtime handle for session context.
        """
        self._adapter = adapter
        self._repo_root = Path(repo_root).resolve()
        self._runtime_handle = runtime_handle

    async def resolve_conflicts(
        self,
        merge_result: MergeResult,
        target_branch: str,
        execution_id: str,
        level_number: int,
    ) -> MergeResolution:
        """Resolve merge conflicts by invoking a Claude agent session.

        This is the main entry point. It:
        1. Validates the input is a CONFLICT MergeResult
        2. Re-starts the merge so conflict markers are present
        3. Builds context and invokes the merge-agent
        4. Verifies the resolution and completes the merge
        5. Returns structured results with warnings

        Args:
            merge_result: A MergeResult with outcome == CONFLICT.
            target_branch: Branch being merged into.
            execution_id: Execution identifier for scope tracking.
            level_number: Level number for scope tracking.

        Returns:
            MergeResolution describing the outcome.
        """
        start_time = datetime.now(UTC)
        ac_index = merge_result.ac_index
        ac_branch = merge_result.ac_branch

        # Fast path: not a conflict — skip
        if merge_result.outcome != MergeOutcome.CONFLICT:
            return MergeResolution(
                ac_index=ac_index,
                ac_branch=ac_branch,
                outcome=MergeResolutionOutcome.SKIPPED,
            )

        log.info(
            "merge_agent.resolution.started",
            ac_index=ac_index,
            ac_branch=ac_branch,
            conflicting_files=list(merge_result.conflicting_files),
        )

        # Step 1: Re-start the merge so conflict markers are in the working tree
        try:
            merge_started = _start_merge_with_conflicts(
                self._repo_root, target_branch, ac_branch,
            )
        except WorktreeError as exc:
            duration = (datetime.now(UTC) - start_time).total_seconds()
            return MergeResolution(
                ac_index=ac_index,
                ac_branch=ac_branch,
                outcome=MergeResolutionOutcome.FAILED,
                error_message=f"Failed to restart merge: {exc}",
                duration_seconds=duration,
            )

        if not merge_started:
            # Auto-merge succeeded on retry — no agent needed
            sha = _run_git(["rev-parse", "HEAD"], self._repo_root)
            duration = (datetime.now(UTC) - start_time).total_seconds()
            return MergeResolution(
                ac_index=ac_index,
                ac_branch=ac_branch,
                outcome=MergeResolutionOutcome.RESOLVED,
                files_resolved=merge_result.conflicting_files,
                merge_sha=sha,
                duration_seconds=duration,
                agent_summary="Auto-merge succeeded on retry without agent intervention.",
            )

        # Step 2: Read conflicting file contents (with conflict markers)
        conflict_contents = _read_conflict_file_contents(
            self._repo_root, merge_result.conflicting_files,
        )

        # Step 3: Build prompt and invoke the merge-agent
        prompt = _build_merge_prompt(merge_result, target_branch, conflict_contents)

        resolution = await self._invoke_agent(
            prompt=prompt,
            merge_result=merge_result,
            target_branch=target_branch,
            execution_id=execution_id,
            level_number=level_number,
            start_time=start_time,
        )

        return resolution

    async def _invoke_agent(
        self,
        prompt: str,
        merge_result: MergeResult,
        target_branch: str,
        execution_id: str,
        level_number: int,
        start_time: datetime,
    ) -> MergeResolution:
        """Invoke the Claude merge-agent and process results.

        The agent runs with Read/Edit/Bash/Grep/Glob tools, operating on
        the working tree where conflict markers are present. After the agent
        completes, we verify resolution and complete the merge.
        """
        ac_index = merge_result.ac_index
        ac_branch = merge_result.ac_branch
        # Build scope for future persistence/audit (not yet wired to runtime handle)
        _build_merge_agent_runtime_scope(execution_id, level_number, ac_index)

        session_id: str | None = None
        final_text = ""
        messages: list[AgentMessage] = []
        runtime_handle = self._runtime_handle

        try:
            async for message in self._adapter.execute_task(
                prompt=prompt,
                tools=MERGE_AGENT_TOOLS,
                system_prompt=MERGE_AGENT_SYSTEM_PROMPT,
                resume_handle=runtime_handle,
            ):
                messages.append(message)
                if message.resume_handle is not None:
                    runtime_handle = message.resume_handle
                if (
                    message.resume_handle is not None
                    and message.resume_handle.native_session_id
                ):
                    session_id = message.resume_handle.native_session_id
                elif message.data.get("session_id"):
                    session_id = message.data["session_id"]
                if message.is_final:
                    final_text = message.content

        except Exception as exc:
            log.exception(
                "merge_agent.session.failed",
                ac_index=ac_index,
                ac_branch=ac_branch,
                error=str(exc),
            )
            _abort_merge(self._repo_root)
            duration = (datetime.now(UTC) - start_time).total_seconds()
            return MergeResolution(
                ac_index=ac_index,
                ac_branch=ac_branch,
                outcome=MergeResolutionOutcome.FAILED,
                error_message=f"Merge-agent session failed: {exc}",
                duration_seconds=duration,
                session_id=session_id,
                messages=tuple(messages),
            )

        # Step 4: Verify the resolution
        resolved_files, remaining_files = _check_remaining_conflicts(
            self._repo_root, merge_result.conflicting_files,
        )

        # Parse agent output for summary and status
        agent_summary, agent_status = _parse_agent_output(final_text)

        # Determine outcome based on actual file state (not just agent's claim)
        if remaining_files:
            # Agent didn't resolve everything
            if resolved_files:
                outcome = MergeResolutionOutcome.PARTIAL
            else:
                outcome = MergeResolutionOutcome.FAILED

            # Abort the merge — can't complete with unresolved conflicts
            _abort_merge(self._repo_root)

            warnings = _generate_resolution_warnings(
                agent_summary, resolved_files, remaining_files,
            )
            duration = (datetime.now(UTC) - start_time).total_seconds()

            log.warning(
                "merge_agent.resolution.incomplete",
                ac_index=ac_index,
                resolved=len(resolved_files),
                remaining=len(remaining_files),
                remaining_files=remaining_files,
            )

            return MergeResolution(
                ac_index=ac_index,
                ac_branch=ac_branch,
                outcome=outcome,
                files_resolved=tuple(resolved_files),
                files_remaining=tuple(remaining_files),
                warnings=tuple(warnings),
                agent_summary=agent_summary,
                duration_seconds=duration,
                session_id=session_id,
                messages=tuple(messages),
            )

        # Step 5: All conflicts resolved — complete the merge commit
        merge_sha = _complete_merge_after_resolution(
            self._repo_root, ac_index, ac_branch,
        )

        if merge_sha is None:
            _abort_merge(self._repo_root)
            duration = (datetime.now(UTC) - start_time).total_seconds()
            return MergeResolution(
                ac_index=ac_index,
                ac_branch=ac_branch,
                outcome=MergeResolutionOutcome.FAILED,
                files_resolved=tuple(resolved_files),
                error_message="Failed to create merge commit after resolving conflicts",
                agent_summary=agent_summary,
                duration_seconds=duration,
                session_id=session_id,
                messages=tuple(messages),
            )

        # Generate warnings for non-trivial resolutions
        warnings = _generate_resolution_warnings(
            agent_summary, resolved_files, remaining_files,
        )

        duration = (datetime.now(UTC) - start_time).total_seconds()

        log.info(
            "merge_agent.resolution.completed",
            ac_index=ac_index,
            ac_branch=ac_branch,
            files_resolved=len(resolved_files),
            merge_sha=merge_sha[:12],
            has_warnings=len(warnings) > 0,
            duration_seconds=round(duration, 2),
        )

        return MergeResolution(
            ac_index=ac_index,
            ac_branch=ac_branch,
            outcome=MergeResolutionOutcome.RESOLVED,
            files_resolved=tuple(resolved_files),
            merge_sha=merge_sha,
            warnings=tuple(warnings),
            agent_summary=agent_summary,
            duration_seconds=duration,
            session_id=session_id,
            messages=tuple(messages),
        )

    async def resolve_all_conflicts(
        self,
        conflict_results: list[MergeResult],
        target_branch: str,
        execution_id: str,
        level_number: int,
    ) -> list[MergeResolution]:
        """Resolve multiple merge conflicts sequentially.

        Conflicts are resolved one at a time in order of ac_index, since
        each resolution may affect the merge state for subsequent branches.

        Args:
            conflict_results: List of MergeResult with CONFLICT outcomes.
            target_branch: Branch being merged into.
            execution_id: Execution identifier.
            level_number: Level number.

        Returns:
            List of MergeResolution results in the same order.
        """
        sorted_results = sorted(conflict_results, key=lambda r: r.ac_index)
        resolutions: list[MergeResolution] = []

        log.info(
            "merge_agent.batch.started",
            conflict_count=len(sorted_results),
            target_branch=target_branch,
        )

        for merge_result in sorted_results:
            resolution = await self.resolve_conflicts(
                merge_result=merge_result,
                target_branch=target_branch,
                execution_id=execution_id,
                level_number=level_number,
            )
            resolutions.append(resolution)

            # If a resolution failed, log but continue with remaining
            if not resolution.resolved:
                log.warning(
                    "merge_agent.batch.resolution_failed",
                    ac_index=merge_result.ac_index,
                    outcome=resolution.outcome.value,
                )

        resolved_count = sum(1 for r in resolutions if r.resolved)
        log.info(
            "merge_agent.batch.completed",
            total=len(sorted_results),
            resolved=resolved_count,
            failed=len(sorted_results) - resolved_count,
        )

        return resolutions


def collect_merge_warnings_for_next_level(
    resolutions: list[MergeResolution],
) -> tuple[str, ...]:
    """Collect all merge-agent warnings into warnings_for_next_level format.

    Aggregates warnings from individual MergeResolution instances into a
    flat tuple suitable for injection into LevelContext.merge_warnings.
    Each warning is prefixed with AC context for traceability.

    Non-trivial resolutions (partial, large scope, keyword-flagged) are
    surfaced so downstream ACs or the evaluation pipeline can verify.

    Args:
        resolutions: List of MergeResolution from merge-agent processing.

    Returns:
        Tuple of warning strings for injection into the next level context.
        Empty tuple when all merges were clean with no warnings.
    """
    warnings: list[str] = []

    for resolution in resolutions:
        if resolution.outcome == MergeResolutionOutcome.SKIPPED:
            continue

        prefix = f"[AC {resolution.ac_index}, branch {resolution.ac_branch}]"

        # Include per-resolution warnings with AC context
        for warning in resolution.warnings:
            warnings.append(f"{prefix} {warning}")

        # Flag partial resolutions explicitly
        if resolution.outcome == MergeResolutionOutcome.PARTIAL:
            remaining = ", ".join(resolution.files_remaining)
            warnings.append(
                f"{prefix} Merge-agent only partially resolved conflicts. "
                f"Unresolved files: {remaining}"
            )

        # Flag failed resolutions
        if resolution.outcome == MergeResolutionOutcome.FAILED:
            error = resolution.error_message or "unknown error"
            warnings.append(
                f"{prefix} Merge-agent failed to resolve conflicts: {error}"
            )

    return tuple(warnings)


__all__ = [
    "MERGE_AGENT_SYSTEM_PROMPT",
    "MERGE_AGENT_TOOLS",
    "MergeAgentInvoker",
    "MergeResolution",
    "MergeResolutionOutcome",
    "collect_merge_warnings_for_next_level",
]
