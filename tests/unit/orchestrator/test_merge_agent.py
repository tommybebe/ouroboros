"""Tests for merge-agent invocation and conflict resolution.

Covers Sub-AC 2 of AC 5: merge-agent invocation function that detects failed
auto-merge, extracts conflict markers/diff from git output, calls the
merge-agent with the conflict context, and applies resolved content back.
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
    _build_merge_prompt,
    _check_remaining_conflicts,
    _complete_merge_after_resolution,
    _generate_resolution_warnings,
    _parse_agent_output,
    _read_conflict_file_contents,
    _start_merge_with_conflicts,
)
from ouroboros.orchestrator.worktree_merge import MergeOutcome, MergeResult

# ---------------------------------------------------------------------------
# Helpers
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
    filename: str,
    target_content: str,
    ac_content: str,
) -> None:
    """Create two branches that conflict on the same file."""
    default_branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo)

    # Create target branch with one version
    _git(["checkout", "-b", target_branch], repo)
    (repo / filename).write_text(target_content)
    _git(["add", filename], repo)
    _git(["commit", "-m", f"Add {filename} on target"], repo)

    # Create AC branch from default (not target) with different version
    _git(["checkout", default_branch], repo)
    _git(["checkout", "-b", ac_branch], repo)
    (repo / filename).write_text(ac_content)
    _git(["add", filename], repo)
    _git(["commit", "-m", f"Add {filename} on AC branch"], repo)

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
    """Mock AgentRuntime that simulates a merge-agent resolving conflicts."""

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

        if self.should_fail:
            raise RuntimeError("Mock agent failure")

        # Simulate the agent resolving files by writing resolved content
        if self.repo_root:
            for filepath, content in self.resolve_files.items():
                full_path = self.repo_root / filepath
                full_path.write_text(content)

        yield MockAgentMessage(
            content=self.final_text or "RESOLUTION_SUMMARY:\nResolved all conflicts.\nRESOLUTION_STATUS: RESOLVED",
            is_final=True,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """Create a test git repo."""
    return _init_repo(tmp_path / "repo")


@pytest.fixture()
def conflict_repo(repo: Path) -> tuple[Path, str, str]:
    """Create a repo with conflicting branches."""
    target = "ooo/test_target"
    ac_branch = "ooo/test_target_ac_0"
    _create_conflicting_branches(
        repo,
        target,
        ac_branch,
        "shared.py",
        "def hello():\n    return 'from target'\n",
        "def hello():\n    return 'from ac'\n",
    )
    return repo, target, ac_branch


# ---------------------------------------------------------------------------
# Unit tests: _build_merge_prompt
# ---------------------------------------------------------------------------


class TestBuildMergePrompt:
    """Tests for prompt construction."""

    def test_includes_branch_names(self) -> None:
        result = MergeResult(
            ac_index=0,
            ac_branch="ooo/test_ac_0",
            outcome=MergeOutcome.CONFLICT,
            conflicting_files=("file.py",),
            conflict_diff="diff --git ...",
        )
        prompt = _build_merge_prompt(
            result, "ooo/test_target", {"file.py": "content"}
        )
        assert "ooo/test_target" in prompt
        assert "ooo/test_ac_0" in prompt

    def test_includes_file_list(self) -> None:
        result = MergeResult(
            ac_index=1,
            ac_branch="ooo/test_ac_1",
            outcome=MergeOutcome.CONFLICT,
            conflicting_files=("a.py", "b.py"),
            conflict_diff="",
        )
        prompt = _build_merge_prompt(result, "target", {"a.py": "a", "b.py": "b"})
        assert "a.py" in prompt
        assert "b.py" in prompt

    def test_includes_file_contents(self) -> None:
        result = MergeResult(
            ac_index=0,
            ac_branch="branch",
            outcome=MergeOutcome.CONFLICT,
            conflicting_files=("test.py",),
            conflict_diff="",
        )
        prompt = _build_merge_prompt(
            result, "target",
            {"test.py": "<<<<<<< HEAD\nours\n=======\ntheirs\n>>>>>>> branch\n"},
        )
        assert "<<<<<<< HEAD" in prompt
        assert "ours" in prompt
        assert "theirs" in prompt

    def test_truncates_large_content(self) -> None:
        result = MergeResult(
            ac_index=0,
            ac_branch="branch",
            outcome=MergeOutcome.CONFLICT,
            conflicting_files=("big.py",),
            conflict_diff="",
        )
        big_content = "x" * 10000
        prompt = _build_merge_prompt(result, "target", {"big.py": big_content})
        assert "(truncated)" in prompt

    def test_truncates_large_diff(self) -> None:
        result = MergeResult(
            ac_index=0,
            ac_branch="branch",
            outcome=MergeOutcome.CONFLICT,
            conflicting_files=("f.py",),
            conflict_diff="d" * 8000,
        )
        prompt = _build_merge_prompt(result, "target", {})
        assert "(diff truncated)" in prompt


# ---------------------------------------------------------------------------
# Unit tests: _check_remaining_conflicts
# ---------------------------------------------------------------------------


class TestCheckRemainingConflicts:
    """Tests for conflict marker detection."""

    def test_all_resolved(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("clean content\n")
        resolved, remaining = _check_remaining_conflicts(tmp_path, ("a.py",))
        assert resolved == ["a.py"]
        assert remaining == []

    def test_conflict_markers_remain(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("<<<<<<< HEAD\nours\n=======\ntheirs\n>>>>>>> branch\n")
        resolved, remaining = _check_remaining_conflicts(tmp_path, ("a.py",))
        assert resolved == []
        assert remaining == ["a.py"]

    def test_partial_resolution(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("clean\n")
        (tmp_path / "b.py").write_text("<<<<<<< HEAD\nx\n=======\ny\n>>>>>>> z\n")
        resolved, remaining = _check_remaining_conflicts(tmp_path, ("a.py", "b.py"))
        assert resolved == ["a.py"]
        assert remaining == ["b.py"]

    def test_missing_file_counts_as_remaining(self, tmp_path: Path) -> None:
        resolved, remaining = _check_remaining_conflicts(tmp_path, ("missing.py",))
        assert resolved == []
        assert remaining == ["missing.py"]


# ---------------------------------------------------------------------------
# Unit tests: _parse_agent_output
# ---------------------------------------------------------------------------


class TestParseAgentOutput:
    """Tests for agent output parsing."""

    def test_parses_structured_output(self) -> None:
        text = (
            "Some work...\n"
            "RESOLUTION_SUMMARY:\nResolved a.py by combining both additions.\n"
            "RESOLUTION_STATUS: RESOLVED"
        )
        summary, status = _parse_agent_output(text)
        assert "a.py" in summary
        assert status == "RESOLVED"

    def test_parses_partial_status(self) -> None:
        text = "RESOLUTION_SUMMARY:\nPartial.\nRESOLUTION_STATUS: PARTIAL"
        summary, status = _parse_agent_output(text)
        assert status == "PARTIAL"

    def test_parses_failed_status(self) -> None:
        text = "RESOLUTION_SUMMARY:\nFailed.\nRESOLUTION_STATUS: FAILED"
        _, status = _parse_agent_output(text)
        assert status == "FAILED"

    def test_fallback_for_unstructured(self) -> None:
        text = "I resolved the conflicts by editing the files."
        summary, status = _parse_agent_output(text)
        assert "resolved" in summary.lower()
        assert status == "RESOLVED"  # default

    def test_empty_input(self) -> None:
        summary, status = _parse_agent_output("")
        assert summary == ""
        assert status == "RESOLVED"


# ---------------------------------------------------------------------------
# Unit tests: _generate_resolution_warnings
# ---------------------------------------------------------------------------


class TestGenerateResolutionWarnings:
    """Tests for warning generation."""

    def test_no_warnings_for_clean_resolution(self) -> None:
        warnings = _generate_resolution_warnings("All good", ["a.py"], [])
        assert len(warnings) == 0

    def test_warns_on_remaining_files(self) -> None:
        warnings = _generate_resolution_warnings("", [], ["b.py"])
        assert any("unresolved" in w.lower() for w in warnings)
        assert any("b.py" in w for w in warnings)

    def test_warns_on_nontrivial_keywords(self) -> None:
        warnings = _generate_resolution_warnings(
            "I chose version A because B was incompatible",
            ["a.py"], [],
        )
        assert any("non-trivial" in w.lower() for w in warnings)

    def test_warns_on_large_conflict_scope(self) -> None:
        warnings = _generate_resolution_warnings(
            "All resolved",
            ["a.py", "b.py", "c.py", "d.py"],
            [],
        )
        assert any("large conflict scope" in w.lower() for w in warnings)


# ---------------------------------------------------------------------------
# Unit tests: _read_conflict_file_contents
# ---------------------------------------------------------------------------


class TestReadConflictFileContents:
    """Tests for reading conflict files."""

    def test_reads_existing_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("content_a")
        (tmp_path / "b.py").write_text("content_b")
        result = _read_conflict_file_contents(tmp_path, ("a.py", "b.py"))
        assert result == {"a.py": "content_a", "b.py": "content_b"}

    def test_skips_missing_files(self, tmp_path: Path) -> None:
        result = _read_conflict_file_contents(tmp_path, ("missing.py",))
        assert result == {}


# ---------------------------------------------------------------------------
# Integration tests: _start_merge_with_conflicts
# ---------------------------------------------------------------------------


class TestStartMergeWithConflicts:
    """Tests for restarting merge with conflict markers."""

    def test_starts_conflict_merge(self, conflict_repo: tuple[Path, str, str]) -> None:
        repo, target, ac_branch = conflict_repo
        started = _start_merge_with_conflicts(repo, target, ac_branch)
        assert started is True

        # Verify conflict markers are present
        content = (repo / "shared.py").read_text()
        assert "<<<<<<< " in content or "=======" in content

        # Clean up
        _git(["merge", "--abort"], repo)

    def test_returns_false_when_auto_merge_succeeds(self, repo: Path) -> None:
        """When branches don't conflict, merge succeeds automatically."""
        default_branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo)

        # Create non-conflicting branches
        _git(["checkout", "-b", "target"], repo)
        (repo / "file_a.py").write_text("a\n")
        _git(["add", "."], repo)
        _git(["commit", "-m", "target change"], repo)

        _git(["checkout", default_branch], repo)
        _git(["checkout", "-b", "ac_branch"], repo)
        (repo / "file_b.py").write_text("b\n")
        _git(["add", "."], repo)
        _git(["commit", "-m", "ac change"], repo)

        _git(["checkout", "target"], repo)

        started = _start_merge_with_conflicts(repo, "target", "ac_branch")
        assert started is False  # Auto-merge succeeded


# ---------------------------------------------------------------------------
# Integration tests: _complete_merge_after_resolution
# ---------------------------------------------------------------------------


class TestCompleteMergeAfterResolution:
    """Tests for completing a merge after agent resolution."""

    def test_commits_resolved_merge(self, conflict_repo: tuple[Path, str, str]) -> None:
        repo, target, ac_branch = conflict_repo

        # Start the conflicting merge
        _start_merge_with_conflicts(repo, target, ac_branch)

        # Simulate agent resolving the conflict
        (repo / "shared.py").write_text("def hello():\n    return 'merged'\n")

        # Complete the merge
        sha = _complete_merge_after_resolution(repo, 0, ac_branch)
        assert sha is not None
        assert len(sha) == 40  # Full SHA

        # Verify the commit exists
        log_output = _git(["log", "--oneline", "-1"], repo)
        assert "agent-resolved" in log_output


# ---------------------------------------------------------------------------
# Integration tests: MergeAgentInvoker
# ---------------------------------------------------------------------------


class TestMergeAgentInvoker:
    """Integration tests for the full merge-agent flow."""

    @pytest.mark.anyio()
    async def test_skips_non_conflict_result(self, repo: Path) -> None:
        adapter = MockAgentRuntime(repo_root=repo)
        invoker = MergeAgentInvoker(adapter=adapter, repo_root=repo)

        result = MergeResult(
            ac_index=0,
            ac_branch="ooo/test_ac_0",
            outcome=MergeOutcome.SUCCESS,
            merge_sha="abc123",
        )
        resolution = await invoker.resolve_conflicts(
            result, "target", "exec_1", 1,
        )
        assert resolution.outcome == MergeResolutionOutcome.SKIPPED
        assert len(adapter.prompts_received) == 0

    @pytest.mark.anyio()
    async def test_resolves_conflict_successfully(
        self, conflict_repo: tuple[Path, str, str],
    ) -> None:
        repo, target, ac_branch = conflict_repo

        adapter = MockAgentRuntime(
            repo_root=repo,
            resolve_files={
                "shared.py": "def hello():\n    return 'merged from both'\n",
            },
            final_text=(
                "RESOLUTION_SUMMARY:\nMerged hello() return values.\n"
                "RESOLUTION_STATUS: RESOLVED"
            ),
        )
        invoker = MergeAgentInvoker(adapter=adapter, repo_root=repo)

        merge_result = MergeResult(
            ac_index=0,
            ac_branch=ac_branch,
            outcome=MergeOutcome.CONFLICT,
            conflicting_files=("shared.py",),
            conflict_diff="diff showing conflict",
        )

        resolution = await invoker.resolve_conflicts(
            merge_result, target, "exec_1", 1,
        )

        assert resolution.resolved
        assert resolution.outcome == MergeResolutionOutcome.RESOLVED
        assert "shared.py" in resolution.files_resolved
        assert resolution.merge_sha is not None
        assert len(resolution.files_remaining) == 0
        assert len(adapter.prompts_received) == 1

    @pytest.mark.anyio()
    async def test_handles_agent_failure(
        self, conflict_repo: tuple[Path, str, str],
    ) -> None:
        repo, target, ac_branch = conflict_repo

        adapter = MockAgentRuntime(
            repo_root=repo,
            should_fail=True,
        )
        invoker = MergeAgentInvoker(adapter=adapter, repo_root=repo)

        merge_result = MergeResult(
            ac_index=0,
            ac_branch=ac_branch,
            outcome=MergeOutcome.CONFLICT,
            conflicting_files=("shared.py",),
            conflict_diff="diff",
        )

        resolution = await invoker.resolve_conflicts(
            merge_result, target, "exec_1", 1,
        )

        assert resolution.outcome == MergeResolutionOutcome.FAILED
        assert "failed" in resolution.error_message.lower()

        # Verify the repo is in a clean state (merge aborted)
        status = _git(["status", "--porcelain"], repo)
        assert "UU" not in status  # No unmerged files

    @pytest.mark.anyio()
    async def test_partial_resolution(
        self, conflict_repo: tuple[Path, str, str],
    ) -> None:
        """When agent resolves some but not all files."""
        repo, target, ac_branch = conflict_repo

        # Add a second conflicting file
        _git(["checkout", target], repo)
        (repo / "other.py").write_text("target version\n")
        _git(["add", "."], repo)
        _git(["commit", "-m", "Add other.py on target"], repo)

        _git(["checkout", ac_branch], repo)
        (repo / "other.py").write_text("ac version\n")
        _git(["add", "."], repo)
        _git(["commit", "-m", "Add other.py on AC"], repo)
        _git(["checkout", target], repo)

        # Agent only resolves shared.py, leaves other.py with markers
        adapter = MockAgentRuntime(
            repo_root=repo,
            resolve_files={
                "shared.py": "def hello():\n    return 'merged'\n",
                # Don't resolve other.py — leave conflict markers
            },
            final_text="RESOLUTION_SUMMARY:\nPartial.\nRESOLUTION_STATUS: PARTIAL",
        )
        invoker = MergeAgentInvoker(adapter=adapter, repo_root=repo)

        merge_result = MergeResult(
            ac_index=0,
            ac_branch=ac_branch,
            outcome=MergeOutcome.CONFLICT,
            conflicting_files=("shared.py", "other.py"),
            conflict_diff="diff",
        )

        resolution = await invoker.resolve_conflicts(
            merge_result, target, "exec_1", 1,
        )

        assert resolution.outcome == MergeResolutionOutcome.PARTIAL
        assert "shared.py" in resolution.files_resolved
        assert "other.py" in resolution.files_remaining

    @pytest.mark.anyio()
    async def test_resolve_all_conflicts(self, repo: Path) -> None:
        """Test batch resolution of multiple conflict results."""
        adapter = MockAgentRuntime(repo_root=repo)
        invoker = MergeAgentInvoker(adapter=adapter, repo_root=repo)

        # Non-conflict results are skipped
        results = [
            MergeResult(
                ac_index=0,
                ac_branch="ooo/ac_0",
                outcome=MergeOutcome.SUCCESS,
            ),
            MergeResult(
                ac_index=1,
                ac_branch="ooo/ac_1",
                outcome=MergeOutcome.NOTHING_TO_MERGE,
            ),
        ]

        resolutions = await invoker.resolve_all_conflicts(
            results, "target", "exec_1", 1,
        )
        assert len(resolutions) == 2
        assert all(r.outcome == MergeResolutionOutcome.SKIPPED for r in resolutions)


# ---------------------------------------------------------------------------
# Unit tests: MergeResolution
# ---------------------------------------------------------------------------


class TestMergeResolution:
    """Tests for MergeResolution data class."""

    def test_resolved_property(self) -> None:
        r = MergeResolution(
            ac_index=0,
            ac_branch="b",
            outcome=MergeResolutionOutcome.RESOLVED,
        )
        assert r.resolved is True

    def test_not_resolved(self) -> None:
        r = MergeResolution(
            ac_index=0,
            ac_branch="b",
            outcome=MergeResolutionOutcome.FAILED,
        )
        assert r.resolved is False

    def test_has_warnings(self) -> None:
        r = MergeResolution(
            ac_index=0,
            ac_branch="b",
            outcome=MergeResolutionOutcome.RESOLVED,
            warnings=("Warning 1",),
        )
        assert r.has_warnings is True

    def test_no_warnings(self) -> None:
        r = MergeResolution(
            ac_index=0,
            ac_branch="b",
            outcome=MergeResolutionOutcome.RESOLVED,
        )
        assert r.has_warnings is False

    def test_to_dict(self) -> None:
        r = MergeResolution(
            ac_index=2,
            ac_branch="ooo/ac_2",
            outcome=MergeResolutionOutcome.RESOLVED,
            files_resolved=("a.py",),
            merge_sha="abc123",
            warnings=("w1",),
        )
        d = r.to_dict()
        assert d["ac_index"] == 2
        assert d["outcome"] == "resolved"
        assert d["files_resolved"] == ["a.py"]
        assert d["merge_sha"] == "abc123"
        assert d["warnings"] == ["w1"]
