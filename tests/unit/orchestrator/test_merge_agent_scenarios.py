"""Scenario tests for merge-agent flow.

Covers Sub-AC 3 of AC 5: merge-agent scenarios including:
- Simple conflict resolution (single file, clean merge)
- Multi-file conflicts (several files with varying conflict types)
- Unresolvable conflicts requiring human escalation

These tests exercise the full MergeAgentInvoker flow with real git repos
and mock agent runtimes that simulate different resolution behaviours.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
from typing import Any

import pytest

from ouroboros.orchestrator.merge_agent import (
    MergeAgentInvoker,
    MergeResolution,
    MergeResolutionOutcome,
    _generate_resolution_warnings,
)
from ouroboros.orchestrator.worktree_merge import MergeOutcome, MergeResult

# ---------------------------------------------------------------------------
# Git helpers
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


def _create_conflicting_branches(
    repo: Path,
    target_branch: str,
    ac_branch: str,
    files: dict[str, tuple[str, str]],
) -> None:
    """Create two branches that conflict on specified files.

    Args:
        repo: Path to the git repository.
        target_branch: Name of the target branch.
        ac_branch: Name of the AC branch.
        files: Mapping of filename to (target_content, ac_content) pairs.
    """
    default_branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo)

    # Create target branch
    _git(["checkout", "-b", target_branch], repo)
    for filename, (target_content, _) in files.items():
        filepath = repo / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(target_content)
    _git(["add", "."], repo)
    _git(["commit", "-m", "Target branch changes"], repo)

    # Create AC branch from default
    _git(["checkout", default_branch], repo)
    _git(["checkout", "-b", ac_branch], repo)
    for filename, (_, ac_content) in files.items():
        filepath = repo / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(ac_content)
    _git(["add", "."], repo)
    _git(["commit", "-m", "AC branch changes"], repo)

    # Go back to target
    _git(["checkout", target_branch], repo)


# ---------------------------------------------------------------------------
# Mock Agent Runtime
# ---------------------------------------------------------------------------


@dataclass
class MockAgentMessage:
    """Minimal AgentMessage for testing."""

    content: str = ""
    is_final: bool = False
    resume_handle: Any = None
    data: dict[str, Any] = field(default_factory=dict)


class MockAgentRuntime:
    """Mock that simulates a merge-agent resolving (or failing to resolve) conflicts."""

    def __init__(
        self,
        *,
        resolve_files: dict[str, str] | None = None,
        final_text: str = "",
        should_fail: bool = False,
        repo_root: Path | None = None,
    ) -> None:
        self.resolve_files = resolve_files or {}
        self.final_text = final_text
        self.should_fail = should_fail
        self.repo_root = repo_root
        self.prompts_received: list[str] = []
        self.invocation_count = 0

    @property
    def runtime_backend(self) -> str:
        return "mock"

    @property
    def working_directory(self) -> str | None:
        return str(self.repo_root) if self.repo_root else None

    @property
    def permission_mode(self) -> str | None:
        return "acceptEdits"

    async def execute_task(
        self,
        prompt: str,
        tools: list[str] | None = None,
        system_prompt: str | None = None,
        resume_handle: Any = None,
        resume_session_id: str | None = None,
    ) -> AsyncIterator[MockAgentMessage]:
        self.prompts_received.append(prompt)
        self.invocation_count += 1

        if self.should_fail:
            raise RuntimeError("Mock agent failure — simulating crash")

        # Simulate the agent resolving files by writing resolved content
        if self.repo_root:
            for filepath, content in self.resolve_files.items():
                full_path = self.repo_root / filepath
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content)

        yield MockAgentMessage(
            content=self.final_text
            or "RESOLUTION_SUMMARY:\nResolved all.\nRESOLUTION_STATUS: RESOLVED",
            is_final=True,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """Create a test git repo."""
    return _init_repo(tmp_path / "repo")


# ---------------------------------------------------------------------------
# Scenario 1: Simple conflict resolution
# ---------------------------------------------------------------------------


class TestSimpleConflictResolution:
    """Single-file conflict where the agent cleanly resolves both sides."""

    @pytest.fixture()
    def simple_conflict(self, repo: Path) -> tuple[Path, str, str]:
        target = "ooo/exec_simple_target"
        ac_branch = "ooo/exec_simple_target_ac_0"
        _create_conflicting_branches(
            repo,
            target,
            ac_branch,
            {
                "utils.py": (
                    "def greet(name):\n    return f'Hello, {name}!'\n",
                    "def greet(name):\n    return f'Hi, {name}!'\n",
                ),
            },
        )
        return repo, target, ac_branch

    @pytest.mark.anyio()
    async def test_resolves_single_file_conflict(
        self, simple_conflict: tuple[Path, str, str],
    ) -> None:
        """Agent resolves a single-file conflict — merge completes, SHA recorded."""
        repo, target, ac_branch = simple_conflict

        adapter = MockAgentRuntime(
            repo_root=repo,
            resolve_files={
                "utils.py": "def greet(name):\n    return f'Hello and Hi, {name}!'\n",
            },
            final_text=(
                "RESOLUTION_SUMMARY:\n"
                "Combined both greeting styles into a single return.\n"
                "RESOLUTION_STATUS: RESOLVED"
            ),
        )
        invoker = MergeAgentInvoker(adapter=adapter, repo_root=repo)

        merge_result = MergeResult(
            ac_index=0,
            ac_branch=ac_branch,
            outcome=MergeOutcome.CONFLICT,
            conflicting_files=("utils.py",),
            conflict_diff="diff showing conflict in utils.py",
        )

        resolution = await invoker.resolve_conflicts(
            merge_result, target, "exec_simple", 1,
        )

        assert resolution.outcome == MergeResolutionOutcome.RESOLVED
        assert resolution.resolved is True
        assert resolution.merge_sha is not None
        assert len(resolution.merge_sha) == 40
        assert "utils.py" in resolution.files_resolved
        assert len(resolution.files_remaining) == 0
        assert resolution.error_message == ""

    @pytest.mark.anyio()
    async def test_resolved_content_persisted_in_repo(
        self, simple_conflict: tuple[Path, str, str],
    ) -> None:
        """After resolution, the merged content is in the working tree."""
        repo, target, ac_branch = simple_conflict
        resolved_content = "def greet(name):\n    return f'Resolved: {name}'\n"

        adapter = MockAgentRuntime(
            repo_root=repo,
            resolve_files={"utils.py": resolved_content},
            final_text="RESOLUTION_SUMMARY:\nFixed.\nRESOLUTION_STATUS: RESOLVED",
        )
        invoker = MergeAgentInvoker(adapter=adapter, repo_root=repo)

        merge_result = MergeResult(
            ac_index=0,
            ac_branch=ac_branch,
            outcome=MergeOutcome.CONFLICT,
            conflicting_files=("utils.py",),
            conflict_diff="diff",
        )

        resolution = await invoker.resolve_conflicts(
            merge_result, target, "exec_simple", 1,
        )

        assert resolution.resolved
        # Verify the file in the repo matches the resolved content
        actual = (repo / "utils.py").read_text()
        assert actual == resolved_content

    @pytest.mark.anyio()
    async def test_prompt_contains_conflict_context(
        self, simple_conflict: tuple[Path, str, str],
    ) -> None:
        """Merge-agent receives prompt with branch names, file list, and diff."""
        repo, target, ac_branch = simple_conflict

        adapter = MockAgentRuntime(
            repo_root=repo,
            resolve_files={"utils.py": "resolved\n"},
            final_text="RESOLUTION_SUMMARY:\nDone.\nRESOLUTION_STATUS: RESOLVED",
        )
        invoker = MergeAgentInvoker(adapter=adapter, repo_root=repo)

        merge_result = MergeResult(
            ac_index=0,
            ac_branch=ac_branch,
            outcome=MergeOutcome.CONFLICT,
            conflicting_files=("utils.py",),
            conflict_diff="diff --git a/utils.py b/utils.py",
        )

        await invoker.resolve_conflicts(merge_result, target, "exec_simple", 1)

        assert len(adapter.prompts_received) == 1
        prompt = adapter.prompts_received[0]
        assert target in prompt
        assert ac_branch in prompt
        assert "utils.py" in prompt

    @pytest.mark.anyio()
    async def test_no_warnings_for_clean_simple_resolution(
        self, simple_conflict: tuple[Path, str, str],
    ) -> None:
        """A clean single-file resolution produces no warnings."""
        repo, target, ac_branch = simple_conflict

        adapter = MockAgentRuntime(
            repo_root=repo,
            resolve_files={"utils.py": "resolved\n"},
            final_text=(
                "RESOLUTION_SUMMARY:\n"
                "Merged both greeting variants.\n"
                "RESOLUTION_STATUS: RESOLVED"
            ),
        )
        invoker = MergeAgentInvoker(adapter=adapter, repo_root=repo)

        merge_result = MergeResult(
            ac_index=0,
            ac_branch=ac_branch,
            outcome=MergeOutcome.CONFLICT,
            conflicting_files=("utils.py",),
            conflict_diff="diff",
        )

        resolution = await invoker.resolve_conflicts(
            merge_result, target, "exec_simple", 1,
        )

        assert resolution.resolved
        assert not resolution.has_warnings

    @pytest.mark.anyio()
    async def test_git_log_shows_agent_resolved_commit(
        self, simple_conflict: tuple[Path, str, str],
    ) -> None:
        """The merge commit message indicates agent resolution."""
        repo, target, ac_branch = simple_conflict

        adapter = MockAgentRuntime(
            repo_root=repo,
            resolve_files={"utils.py": "resolved\n"},
            final_text="RESOLUTION_SUMMARY:\nDone.\nRESOLUTION_STATUS: RESOLVED",
        )
        invoker = MergeAgentInvoker(adapter=adapter, repo_root=repo)

        merge_result = MergeResult(
            ac_index=0,
            ac_branch=ac_branch,
            outcome=MergeOutcome.CONFLICT,
            conflicting_files=("utils.py",),
            conflict_diff="diff",
        )

        resolution = await invoker.resolve_conflicts(
            merge_result, target, "exec_simple", 1,
        )

        assert resolution.merge_sha is not None
        log_msg = _git(["log", "--oneline", "-1"], repo)
        assert "agent-resolved" in log_msg


# ---------------------------------------------------------------------------
# Scenario 2: Multi-file conflicts
# ---------------------------------------------------------------------------


class TestMultiFileConflicts:
    """Conflicts spanning multiple files — all resolved, partially resolved, or mixed."""

    @pytest.fixture()
    def multi_conflict(self, repo: Path) -> tuple[Path, str, str]:
        target = "ooo/exec_multi_target"
        ac_branch = "ooo/exec_multi_target_ac_1"
        _create_conflicting_branches(
            repo,
            target,
            ac_branch,
            {
                "models.py": (
                    "class User:\n    role = 'viewer'\n",
                    "class User:\n    role = 'editor'\n",
                ),
                "views.py": (
                    "def index():\n    return render('home.html')\n",
                    "def index():\n    return render('dashboard.html')\n",
                ),
                "config.py": (
                    "DEBUG = False\nLOG_LEVEL = 'WARNING'\n",
                    "DEBUG = True\nLOG_LEVEL = 'DEBUG'\n",
                ),
            },
        )
        return repo, target, ac_branch

    @pytest.mark.anyio()
    async def test_all_files_resolved(
        self, multi_conflict: tuple[Path, str, str],
    ) -> None:
        """Agent resolves all three conflicting files — full RESOLVED outcome."""
        repo, target, ac_branch = multi_conflict

        adapter = MockAgentRuntime(
            repo_root=repo,
            resolve_files={
                "models.py": "class User:\n    role = 'editor'  # upgraded\n",
                "views.py": "def index():\n    return render('dashboard.html')\n",
                "config.py": "DEBUG = False\nLOG_LEVEL = 'DEBUG'\n",
            },
            final_text=(
                "RESOLUTION_SUMMARY:\n"
                "- models.py: kept editor role from AC\n"
                "- views.py: kept dashboard template from AC\n"
                "- config.py: kept DEBUG=False from target, LOG_LEVEL=DEBUG from AC\n"
                "RESOLUTION_STATUS: RESOLVED"
            ),
        )
        invoker = MergeAgentInvoker(adapter=adapter, repo_root=repo)

        merge_result = MergeResult(
            ac_index=1,
            ac_branch=ac_branch,
            outcome=MergeOutcome.CONFLICT,
            conflicting_files=("models.py", "views.py", "config.py"),
            conflict_diff="diff showing three-file conflict",
        )

        resolution = await invoker.resolve_conflicts(
            merge_result, target, "exec_multi", 2,
        )

        assert resolution.outcome == MergeResolutionOutcome.RESOLVED
        assert set(resolution.files_resolved) == {"models.py", "views.py", "config.py"}
        assert len(resolution.files_remaining) == 0
        assert resolution.merge_sha is not None

    @pytest.mark.anyio()
    async def test_partial_resolution_two_of_three(
        self, multi_conflict: tuple[Path, str, str],
    ) -> None:
        """Agent resolves 2 of 3 files — PARTIAL outcome with remaining listed."""
        repo, target, ac_branch = multi_conflict

        adapter = MockAgentRuntime(
            repo_root=repo,
            resolve_files={
                # Resolve models.py and views.py, but leave config.py with markers
                "models.py": "class User:\n    role = 'editor'\n",
                "views.py": "def index():\n    return render('dashboard.html')\n",
                # config.py intentionally NOT resolved — conflict markers remain
            },
            final_text="RESOLUTION_SUMMARY:\nPartial fix.\nRESOLUTION_STATUS: PARTIAL",
        )
        invoker = MergeAgentInvoker(adapter=adapter, repo_root=repo)

        merge_result = MergeResult(
            ac_index=1,
            ac_branch=ac_branch,
            outcome=MergeOutcome.CONFLICT,
            conflicting_files=("models.py", "views.py", "config.py"),
            conflict_diff="diff",
        )

        resolution = await invoker.resolve_conflicts(
            merge_result, target, "exec_multi", 2,
        )

        assert resolution.outcome == MergeResolutionOutcome.PARTIAL
        assert "models.py" in resolution.files_resolved
        assert "views.py" in resolution.files_resolved
        assert "config.py" in resolution.files_remaining
        assert resolution.merge_sha is None  # merge aborted — incomplete

    @pytest.mark.anyio()
    async def test_large_scope_warning_for_many_files(
        self, multi_conflict: tuple[Path, str, str],
    ) -> None:
        """Resolving 4+ files triggers a large-scope warning."""
        repo, target, ac_branch = multi_conflict

        # Add a 4th conflicting file on both branches
        _git(["checkout", target], repo)
        (repo / "extra.py").write_text("target_extra\n")
        _git(["add", "extra.py"], repo)
        _git(["commit", "-m", "add extra on target"], repo)

        _git(["checkout", ac_branch], repo)
        (repo / "extra.py").write_text("ac_extra\n")
        _git(["add", "extra.py"], repo)
        _git(["commit", "-m", "add extra on ac"], repo)
        _git(["checkout", target], repo)

        adapter = MockAgentRuntime(
            repo_root=repo,
            resolve_files={
                "models.py": "resolved_models\n",
                "views.py": "resolved_views\n",
                "config.py": "resolved_config\n",
                "extra.py": "resolved_extra\n",
            },
            final_text=(
                "RESOLUTION_SUMMARY:\nResolved all four files.\n"
                "RESOLUTION_STATUS: RESOLVED"
            ),
        )
        invoker = MergeAgentInvoker(adapter=adapter, repo_root=repo)

        merge_result = MergeResult(
            ac_index=1,
            ac_branch=ac_branch,
            outcome=MergeOutcome.CONFLICT,
            conflicting_files=("models.py", "views.py", "config.py", "extra.py"),
            conflict_diff="diff",
        )

        resolution = await invoker.resolve_conflicts(
            merge_result, target, "exec_multi", 2,
        )

        assert resolution.resolved
        assert resolution.has_warnings
        assert any("large conflict scope" in w.lower() for w in resolution.warnings)

    @pytest.mark.anyio()
    async def test_multi_file_no_files_resolved_is_failed(
        self, multi_conflict: tuple[Path, str, str],
    ) -> None:
        """Agent touches nothing — all files remain conflicted → FAILED."""
        repo, target, ac_branch = multi_conflict

        adapter = MockAgentRuntime(
            repo_root=repo,
            resolve_files={},  # Agent does nothing
            final_text="RESOLUTION_SUMMARY:\nCould not resolve.\nRESOLUTION_STATUS: FAILED",
        )
        invoker = MergeAgentInvoker(adapter=adapter, repo_root=repo)

        merge_result = MergeResult(
            ac_index=1,
            ac_branch=ac_branch,
            outcome=MergeOutcome.CONFLICT,
            conflicting_files=("models.py", "views.py", "config.py"),
            conflict_diff="diff",
        )

        resolution = await invoker.resolve_conflicts(
            merge_result, target, "exec_multi", 2,
        )

        assert resolution.outcome == MergeResolutionOutcome.FAILED
        assert not resolution.resolved
        assert len(resolution.files_resolved) == 0
        # All three should still be unresolved
        assert set(resolution.files_remaining) == {"models.py", "views.py", "config.py"}

    @pytest.mark.anyio()
    async def test_sequential_batch_resolution(self, repo: Path) -> None:
        """resolve_all_conflicts processes multiple ACs sequentially."""
        target = "ooo/exec_batch_target"
        ac0 = "ooo/exec_batch_target_ac_0"
        ac1 = "ooo/exec_batch_target_ac_1"

        # AC 0 conflict
        _create_conflicting_branches(
            repo, target, ac0,
            {"alpha.py": ("target_alpha\n", "ac0_alpha\n")},
        )
        # AC 1 conflict — branch off the same default branch
        _git(
            ["rev-parse", "--abbrev-ref", "HEAD"],
            repo,
        )
        # Need to branch ac1 from the initial commit too
        initial = _git(["rev-list", "--max-parents=0", "HEAD"], repo)
        _git(["checkout", initial], repo)
        _git(["checkout", "-b", ac1], repo)
        (repo / "beta.py").write_text("ac1_beta\n")
        _git(["add", "."], repo)
        _git(["commit", "-m", "AC 1 changes"], repo)

        # Add beta.py to target too so it conflicts
        _git(["checkout", target], repo)
        (repo / "beta.py").write_text("target_beta\n")
        _git(["add", "."], repo)
        _git(["commit", "-m", "target beta"], repo)

        adapter = MockAgentRuntime(
            repo_root=repo,
            resolve_files={
                "alpha.py": "resolved_alpha\n",
                "beta.py": "resolved_beta\n",
            },
            final_text="RESOLUTION_SUMMARY:\nResolved.\nRESOLUTION_STATUS: RESOLVED",
        )
        invoker = MergeAgentInvoker(adapter=adapter, repo_root=repo)

        results = [
            MergeResult(
                ac_index=0,
                ac_branch=ac0,
                outcome=MergeOutcome.CONFLICT,
                conflicting_files=("alpha.py",),
                conflict_diff="diff",
            ),
            MergeResult(
                ac_index=1,
                ac_branch=ac1,
                outcome=MergeOutcome.CONFLICT,
                conflicting_files=("beta.py",),
                conflict_diff="diff",
            ),
        ]

        resolutions = await invoker.resolve_all_conflicts(
            results, target, "exec_batch", 1,
        )

        assert len(resolutions) == 2
        # At least the first one should succeed
        assert resolutions[0].outcome == MergeResolutionOutcome.RESOLVED
        assert adapter.invocation_count >= 1


# ---------------------------------------------------------------------------
# Scenario 3: Unresolvable conflicts — human escalation
# ---------------------------------------------------------------------------


class TestUnresolvableConflicts:
    """Conflicts the agent cannot resolve, requiring human escalation."""

    @pytest.fixture()
    def hard_conflict(self, repo: Path) -> tuple[Path, str, str]:
        """Create a conflict with semantically incompatible changes."""
        target = "ooo/exec_hard_target"
        ac_branch = "ooo/exec_hard_target_ac_0"
        _create_conflicting_branches(
            repo,
            target,
            ac_branch,
            {
                "schema.py": (
                    (
                        "class Schema:\n"
                        "    version = 2\n"
                        "    fields = ['name', 'email']\n"
                        "    def validate(self):\n"
                        "        return len(self.fields) == 2\n"
                    ),
                    (
                        "class Schema:\n"
                        "    version = 3\n"
                        "    fields = ['name', 'email', 'phone']\n"
                        "    def validate(self):\n"
                        "        return len(self.fields) == 3\n"
                    ),
                ),
            },
        )
        return repo, target, ac_branch

    @pytest.mark.anyio()
    async def test_agent_crash_aborts_merge_cleanly(
        self, hard_conflict: tuple[Path, str, str],
    ) -> None:
        """Agent session crashes — merge is aborted, repo left clean."""
        repo, target, ac_branch = hard_conflict

        adapter = MockAgentRuntime(repo_root=repo, should_fail=True)
        invoker = MergeAgentInvoker(adapter=adapter, repo_root=repo)

        merge_result = MergeResult(
            ac_index=0,
            ac_branch=ac_branch,
            outcome=MergeOutcome.CONFLICT,
            conflicting_files=("schema.py",),
            conflict_diff="diff",
        )

        resolution = await invoker.resolve_conflicts(
            merge_result, target, "exec_hard", 1,
        )

        assert resolution.outcome == MergeResolutionOutcome.FAILED
        assert "failed" in resolution.error_message.lower()

        # Verify repo is clean — no in-progress merge
        status = _git(["status", "--porcelain"], repo)
        assert "UU" not in status

    @pytest.mark.anyio()
    async def test_agent_leaves_markers_is_failed(
        self, hard_conflict: tuple[Path, str, str],
    ) -> None:
        """Agent writes output but conflict markers remain — FAILED, not PARTIAL."""
        repo, target, ac_branch = hard_conflict

        adapter = MockAgentRuntime(
            repo_root=repo,
            resolve_files={},  # Doesn't resolve anything
            final_text=(
                "RESOLUTION_SUMMARY:\n"
                "The conflicts are semantically incompatible. "
                "Schema version 2 validates 2 fields but version 3 validates 3 fields. "
                "Cannot combine without breaking one.\n"
                "RESOLUTION_STATUS: FAILED"
            ),
        )
        invoker = MergeAgentInvoker(adapter=adapter, repo_root=repo)

        merge_result = MergeResult(
            ac_index=0,
            ac_branch=ac_branch,
            outcome=MergeOutcome.CONFLICT,
            conflicting_files=("schema.py",),
            conflict_diff="diff",
        )

        resolution = await invoker.resolve_conflicts(
            merge_result, target, "exec_hard", 1,
        )

        assert resolution.outcome == MergeResolutionOutcome.FAILED
        assert not resolution.resolved
        assert "schema.py" in resolution.files_remaining

    @pytest.mark.anyio()
    async def test_failed_resolution_includes_agent_summary(
        self, hard_conflict: tuple[Path, str, str],
    ) -> None:
        """Failed resolution captures the agent's explanation for human review."""
        repo, target, ac_branch = hard_conflict

        explanation = (
            "The schema version and field count validation are tightly coupled. "
            "Cannot merge without choosing one side."
        )

        adapter = MockAgentRuntime(
            repo_root=repo,
            resolve_files={},
            final_text=(
                f"RESOLUTION_SUMMARY:\n{explanation}\n"
                "RESOLUTION_STATUS: FAILED"
            ),
        )
        invoker = MergeAgentInvoker(adapter=adapter, repo_root=repo)

        merge_result = MergeResult(
            ac_index=0,
            ac_branch=ac_branch,
            outcome=MergeOutcome.CONFLICT,
            conflicting_files=("schema.py",),
            conflict_diff="diff",
        )

        resolution = await invoker.resolve_conflicts(
            merge_result, target, "exec_hard", 1,
        )

        assert not resolution.resolved
        # Agent summary preserved for human escalation review
        assert "schema version" in resolution.agent_summary.lower()
        assert "tightly coupled" in resolution.agent_summary.lower()

    @pytest.mark.anyio()
    async def test_nontrivial_resolution_flagged_with_warning(
        self, hard_conflict: tuple[Path, str, str],
    ) -> None:
        """When agent discards one side, warning flags it for human review."""
        repo, target, ac_branch = hard_conflict

        adapter = MockAgentRuntime(
            repo_root=repo,
            resolve_files={
                "schema.py": (
                    "class Schema:\n"
                    "    version = 3\n"
                    "    fields = ['name', 'email', 'phone']\n"
                    "    def validate(self):\n"
                    "        return len(self.fields) == 3\n"
                ),
            },
            final_text=(
                "RESOLUTION_SUMMARY:\n"
                "I chose version 3 because it is more complete. "
                "Discarded version 2 validation logic.\n"
                "RESOLUTION_STATUS: RESOLVED"
            ),
        )
        invoker = MergeAgentInvoker(adapter=adapter, repo_root=repo)

        merge_result = MergeResult(
            ac_index=0,
            ac_branch=ac_branch,
            outcome=MergeOutcome.CONFLICT,
            conflicting_files=("schema.py",),
            conflict_diff="diff",
        )

        resolution = await invoker.resolve_conflicts(
            merge_result, target, "exec_hard", 1,
        )

        assert resolution.resolved
        assert resolution.has_warnings
        # Should flag the non-trivial "chose"/"discarded" keywords
        assert any("non-trivial" in w.lower() for w in resolution.warnings)

    @pytest.mark.anyio()
    async def test_failed_resolution_aborts_merge_state(
        self, hard_conflict: tuple[Path, str, str],
    ) -> None:
        """After failed resolution, git is not in a merge state."""
        repo, target, ac_branch = hard_conflict

        adapter = MockAgentRuntime(
            repo_root=repo,
            resolve_files={},
            final_text="RESOLUTION_SUMMARY:\nFailed.\nRESOLUTION_STATUS: FAILED",
        )
        invoker = MergeAgentInvoker(adapter=adapter, repo_root=repo)

        merge_result = MergeResult(
            ac_index=0,
            ac_branch=ac_branch,
            outcome=MergeOutcome.CONFLICT,
            conflicting_files=("schema.py",),
            conflict_diff="diff",
        )

        await invoker.resolve_conflicts(merge_result, target, "exec_hard", 1)

        # Git should NOT be in merge state after failure
        merge_head = repo / ".git" / "MERGE_HEAD"
        assert not merge_head.exists(), "MERGE_HEAD should not exist after failed resolution"

    @pytest.mark.anyio()
    async def test_batch_continues_after_unresolvable(self, repo: Path) -> None:
        """resolve_all_conflicts does not stop on first failure — processes all."""
        target = "ooo/exec_escalate_target"
        ac0 = "ooo/exec_escalate_target_ac_0"
        ac1 = "ooo/exec_escalate_target_ac_1"

        # Create AC 0 conflict
        _create_conflicting_branches(
            repo, target, ac0,
            {"hard.py": ("target_hard\n", "ac0_hard\n")},
        )

        # Create AC 1 conflict
        initial = _git(["rev-list", "--max-parents=0", "HEAD"], repo)
        _git(["checkout", initial], repo)
        _git(["checkout", "-b", ac1], repo)
        (repo / "easy.py").write_text("ac1_easy\n")
        _git(["add", "."], repo)
        _git(["commit", "-m", "AC 1"], repo)

        _git(["checkout", target], repo)
        (repo / "easy.py").write_text("target_easy\n")
        _git(["add", "."], repo)
        _git(["commit", "-m", "target easy"], repo)

        # Agent fails on first, succeeds on second
        call_count = 0

        class FailThenSucceedRuntime(MockAgentRuntime):
            async def execute_task(self, prompt, **kwargs):  # type: ignore[override]
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("Unresolvable conflict")
                # Resolve the second one
                if self.repo_root:
                    (self.repo_root / "easy.py").write_text("resolved_easy\n")
                yield MockAgentMessage(
                    content="RESOLUTION_SUMMARY:\nResolved.\nRESOLUTION_STATUS: RESOLVED",
                    is_final=True,
                )

        adapter = FailThenSucceedRuntime(repo_root=repo)
        invoker = MergeAgentInvoker(adapter=adapter, repo_root=repo)

        results = [
            MergeResult(
                ac_index=0,
                ac_branch=ac0,
                outcome=MergeOutcome.CONFLICT,
                conflicting_files=("hard.py",),
                conflict_diff="diff",
            ),
            MergeResult(
                ac_index=1,
                ac_branch=ac1,
                outcome=MergeOutcome.CONFLICT,
                conflicting_files=("easy.py",),
                conflict_diff="diff",
            ),
        ]

        resolutions = await invoker.resolve_all_conflicts(
            results, target, "exec_escalate", 1,
        )

        assert len(resolutions) == 2
        assert resolutions[0].outcome == MergeResolutionOutcome.FAILED
        # Second may resolve or fail depending on git state, but it was attempted
        assert call_count == 2

    @pytest.mark.anyio()
    async def test_to_dict_captures_escalation_info(
        self, hard_conflict: tuple[Path, str, str],
    ) -> None:
        """Serialized resolution preserves enough info for human escalation."""
        repo, target, ac_branch = hard_conflict

        adapter = MockAgentRuntime(
            repo_root=repo,
            resolve_files={},
            final_text=(
                "RESOLUTION_SUMMARY:\n"
                "Incompatible schema changes need human decision.\n"
                "RESOLUTION_STATUS: FAILED"
            ),
        )
        invoker = MergeAgentInvoker(adapter=adapter, repo_root=repo)

        merge_result = MergeResult(
            ac_index=0,
            ac_branch=ac_branch,
            outcome=MergeOutcome.CONFLICT,
            conflicting_files=("schema.py",),
            conflict_diff="diff",
        )

        resolution = await invoker.resolve_conflicts(
            merge_result, target, "exec_hard", 1,
        )

        d = resolution.to_dict()
        assert d["outcome"] == "failed"
        assert d["ac_index"] == 0
        assert d["ac_branch"] == ac_branch
        assert "schema.py" in d["files_remaining"]
        assert "Incompatible" in d["agent_summary"]
        assert d["merge_sha"] is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestMergeAgentEdgeCases:
    """Edge cases and boundary conditions for the merge-agent flow."""

    @pytest.mark.anyio()
    async def test_non_conflict_result_skipped(self, repo: Path) -> None:
        """SUCCESS/NOTHING_TO_MERGE results are skipped without agent invocation."""
        adapter = MockAgentRuntime(repo_root=repo)
        invoker = MergeAgentInvoker(adapter=adapter, repo_root=repo)

        for outcome in (MergeOutcome.SUCCESS, MergeOutcome.NOTHING_TO_MERGE, MergeOutcome.ERROR):
            result = MergeResult(
                ac_index=0,
                ac_branch="ooo/test_ac_0",
                outcome=outcome,
            )
            resolution = await invoker.resolve_conflicts(
                result, "target", "exec_edge", 1,
            )
            assert resolution.outcome == MergeResolutionOutcome.SKIPPED

        assert adapter.invocation_count == 0

    @pytest.mark.anyio()
    async def test_resolution_records_duration(self, repo: Path) -> None:
        """All resolutions (even skipped) record a duration."""
        adapter = MockAgentRuntime(repo_root=repo)
        invoker = MergeAgentInvoker(adapter=adapter, repo_root=repo)

        result = MergeResult(
            ac_index=0,
            ac_branch="ooo/test_ac_0",
            outcome=MergeOutcome.SUCCESS,
        )
        resolution = await invoker.resolve_conflicts(
            result, "target", "exec_edge", 1,
        )

        # Skipped resolutions have 0 duration, which is fine
        assert resolution.duration_seconds >= 0.0

    def test_generate_warnings_unresolved_files_escalation(self) -> None:
        """Unresolved files produce a clear escalation warning."""
        warnings = _generate_resolution_warnings(
            "Could not figure it out",
            [],
            ["a.py", "b.py"],
        )
        assert any("unresolved" in w.lower() for w in warnings)
        assert any("a.py" in w and "b.py" in w for w in warnings)

    def test_generate_warnings_incompatible_keyword(self) -> None:
        """Agent mentioning 'incompatible' triggers non-trivial warning."""
        warnings = _generate_resolution_warnings(
            "Changes are incompatible — chose target version",
            ["schema.py"],
            [],
        )
        assert any("non-trivial" in w.lower() for w in warnings)

    def test_resolution_immutability(self) -> None:
        """MergeResolution is frozen — fields cannot be mutated."""
        r = MergeResolution(
            ac_index=0,
            ac_branch="b",
            outcome=MergeResolutionOutcome.FAILED,
            files_remaining=("a.py",),
        )
        with pytest.raises(AttributeError):
            r.outcome = MergeResolutionOutcome.RESOLVED  # type: ignore[misc]
