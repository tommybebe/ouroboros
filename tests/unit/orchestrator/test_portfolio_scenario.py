"""Portfolio.typ scenario: 6 projects, 3 parallel ACs, one shared file.

AC 7: Re-running the portfolio.typ scenario results in all ACs' changes
present with no data loss.

This scenario simulates a common real-world case: a portfolio document
(portfolio.typ) that contains sections for 6 projects, and 3 parallel
ACs each need to update a subset of project descriptions in that same file.

Without worktree isolation, concurrent writes to portfolio.typ cause
data loss (last writer wins) or Edit failures. With isolation, each AC
works on its own worktree branch, and the merge step preserves all changes.

The test validates the full pipeline:
1. File overlap prediction correctly identifies all 3 ACs as overlapping
2. Each AC gets its own worktree branch
3. Each AC independently modifies portfolio.typ (different sections)
4. WorktreeMerger auto-merges non-overlapping hunks successfully
5. When hunks overlap, conflict info is captured for merge-agent dispatch
6. Final file contains ALL ACs' changes with zero data loss
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import subprocess

import pytest

from ouroboros.orchestrator.ac_isolation import (
    classify_isolation_modes,
    needs_worktree,
)
from ouroboros.orchestrator.ac_worktree import ACWorktreeInfo, ACWorktreeManager
from ouroboros.orchestrator.dependency_analyzer import ACDependencySpec
from ouroboros.orchestrator.file_overlap_predictor import predict_file_overlaps
from ouroboros.orchestrator.worktree_merge import WorktreeMerger

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
    (path / "README.md").write_text("# Portfolio Project\n")
    _git(["add", "."], path)
    _git(["commit", "-m", "Initial commit"], path)
    return path


# ---------------------------------------------------------------------------
# Portfolio document template
# ---------------------------------------------------------------------------

# Simulates a Typst portfolio document with 6 project sections.
# Each AC is responsible for updating 2 project descriptions.
PORTFOLIO_TEMPLATE = """\
#set page(paper: "a4")
#set text(font: "Inter", size: 11pt)

= Portfolio

== Project 1: Web Dashboard
Status: TODO
Description: Placeholder for project 1.

== Project 2: Mobile App
Status: TODO
Description: Placeholder for project 2.

== Project 3: API Gateway
Status: TODO
Description: Placeholder for project 3.

== Project 4: Data Pipeline
Status: TODO
Description: Placeholder for project 4.

== Project 5: ML Service
Status: TODO
Description: Placeholder for project 5.

== Project 6: CLI Tool
Status: TODO
Description: Placeholder for project 6.
"""

# AC 0 updates projects 1 & 2
AC0_CONTENT = """\
#set page(paper: "a4")
#set text(font: "Inter", size: 11pt)

= Portfolio

== Project 1: Web Dashboard
Status: Complete
Description: Built a real-time dashboard with React and WebSockets.

== Project 2: Mobile App
Status: Complete
Description: Cross-platform mobile app using Flutter with offline sync.

== Project 3: API Gateway
Status: TODO
Description: Placeholder for project 3.

== Project 4: Data Pipeline
Status: TODO
Description: Placeholder for project 4.

== Project 5: ML Service
Status: TODO
Description: Placeholder for project 5.

== Project 6: CLI Tool
Status: TODO
Description: Placeholder for project 6.
"""

# AC 1 updates projects 3 & 4
AC1_CONTENT = """\
#set page(paper: "a4")
#set text(font: "Inter", size: 11pt)

= Portfolio

== Project 1: Web Dashboard
Status: TODO
Description: Placeholder for project 1.

== Project 2: Mobile App
Status: TODO
Description: Placeholder for project 2.

== Project 3: API Gateway
Status: Complete
Description: High-throughput API gateway with rate limiting and caching.

== Project 4: Data Pipeline
Status: Complete
Description: Streaming data pipeline processing 1M events/sec with Kafka.

== Project 5: ML Service
Status: TODO
Description: Placeholder for project 5.

== Project 6: CLI Tool
Status: TODO
Description: Placeholder for project 6.
"""

# AC 2 updates projects 5 & 6
AC2_CONTENT = """\
#set page(paper: "a4")
#set text(font: "Inter", size: 11pt)

= Portfolio

== Project 1: Web Dashboard
Status: TODO
Description: Placeholder for project 1.

== Project 2: Mobile App
Status: TODO
Description: Placeholder for project 2.

== Project 3: API Gateway
Status: TODO
Description: Placeholder for project 3.

== Project 4: Data Pipeline
Status: TODO
Description: Placeholder for project 4.

== Project 5: ML Service
Status: Complete
Description: ML inference service with auto-scaling and A/B testing.

== Project 6: CLI Tool
Status: Complete
Description: Developer CLI tool with plugin system and shell completions.
"""

# Expected final state: ALL 6 projects updated
EXPECTED_STRINGS = [
    # AC 0's contributions (projects 1 & 2)
    "Built a real-time dashboard with React and WebSockets",
    "Cross-platform mobile app using Flutter with offline sync",
    # AC 1's contributions (projects 3 & 4)
    "High-throughput API gateway with rate limiting and caching",
    "Streaming data pipeline processing 1M events/sec with Kafka",
    # AC 2's contributions (projects 5 & 6)
    "ML inference service with auto-scaling and A/B testing",
    "Developer CLI tool with plugin system and shell completions",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def portfolio_repo(tmp_path: Path) -> Path:
    """Create a test repo with the portfolio.typ base document."""
    repo = _init_repo(tmp_path / "portfolio-repo")
    (repo / "portfolio.typ").write_text(PORTFOLIO_TEMPLATE)
    _git(["add", "portfolio.typ"], repo)
    _git(["commit", "-m", "Add portfolio template"], repo)
    return repo


@pytest.fixture()
def default_branch(portfolio_repo: Path) -> str:
    """Get the default branch name of the portfolio repo."""
    return _git(["rev-parse", "--abbrev-ref", "HEAD"], portfolio_repo)


# ---------------------------------------------------------------------------
# Core scenario tests
# ---------------------------------------------------------------------------


class TestPortfolioScenarioDataLoss:
    """AC 7: Verify the portfolio scenario preserves all ACs' changes."""

    def test_three_acs_non_overlapping_hunks_auto_merge(
        self, portfolio_repo: Path, default_branch: str
    ) -> None:
        """3 ACs editing different sections of portfolio.typ auto-merge cleanly.

        This is the happy path: each AC edits distinct sections (different
        project entries) so git's 3-way merge resolves without conflicts.
        All 6 project descriptions must be present in the final file.
        """
        repo = portfolio_repo

        # Simulate AC 0: updates projects 1 & 2
        _git(["checkout", "-b", "ooo/test_ac_0", default_branch], repo)
        (repo / "portfolio.typ").write_text(AC0_CONTENT)
        _git(["add", "portfolio.typ"], repo)
        _git(["commit", "-m", "AC 0: Update projects 1 & 2"], repo)
        _git(["checkout", default_branch], repo)

        # Simulate AC 1: updates projects 3 & 4
        _git(["checkout", "-b", "ooo/test_ac_1", default_branch], repo)
        (repo / "portfolio.typ").write_text(AC1_CONTENT)
        _git(["add", "portfolio.typ"], repo)
        _git(["commit", "-m", "AC 1: Update projects 3 & 4"], repo)
        _git(["checkout", default_branch], repo)

        # Simulate AC 2: updates projects 5 & 6
        _git(["checkout", "-b", "ooo/test_ac_2", default_branch], repo)
        (repo / "portfolio.typ").write_text(AC2_CONTENT)
        _git(["add", "portfolio.typ"], repo)
        _git(["commit", "-m", "AC 2: Update projects 5 & 6"], repo)
        _git(["checkout", default_branch], repo)

        # Merge all AC branches sequentially
        merger = WorktreeMerger(repo_root=repo)
        plan = merger.merge_all(
            target_branch=default_branch,
            ac_branches=[
                (0, "ooo/test_ac_0"),
                (1, "ooo/test_ac_1"),
                (2, "ooo/test_ac_2"),
            ],
        )

        # All 3 should auto-merge (non-overlapping hunks in same file)
        assert plan.all_succeeded, (
            f"Expected all merges to succeed, got: "
            f"success={plan.success_count}, conflict={plan.conflict_count}, "
            f"results={[r.outcome.value for r in plan.results]}"
        )
        assert plan.success_count == 3
        assert plan.conflict_count == 0

        # Verify ALL 6 project descriptions are present — zero data loss
        final_content = (repo / "portfolio.typ").read_text()
        for expected in EXPECTED_STRINGS:
            assert expected in final_content, (
                f"Missing AC change in final file: '{expected}'"
            )

        # Verify no TODO placeholders remain for the 6 projects
        # (all should have been replaced by actual descriptions)
        assert final_content.count("Status: Complete") == 6
        assert "Placeholder for project" not in final_content

    def test_merge_preserves_non_portfolio_files(
        self, portfolio_repo: Path, default_branch: str
    ) -> None:
        """Merging portfolio.typ changes doesn't affect other files."""
        repo = portfolio_repo

        # Add an unrelated file to the base
        (repo / "config.yaml").write_text("version: 1\n")
        _git(["add", "config.yaml"], repo)
        _git(["commit", "-m", "Add config"], repo)

        # AC 0 modifies portfolio.typ AND adds its own file
        _git(["checkout", "-b", "ooo/test_ac_0", default_branch], repo)
        (repo / "portfolio.typ").write_text(AC0_CONTENT)
        (repo / "notes_ac0.txt").write_text("AC 0 notes\n")
        _git(["add", "."], repo)
        _git(["commit", "-m", "AC 0 changes"], repo)
        _git(["checkout", default_branch], repo)

        # AC 1 modifies portfolio.typ AND adds its own file
        _git(["checkout", "-b", "ooo/test_ac_1", default_branch], repo)
        (repo / "portfolio.typ").write_text(AC1_CONTENT)
        (repo / "notes_ac1.txt").write_text("AC 1 notes\n")
        _git(["add", "."], repo)
        _git(["commit", "-m", "AC 1 changes"], repo)
        _git(["checkout", default_branch], repo)

        merger = WorktreeMerger(repo_root=repo)
        plan = merger.merge_all(
            target_branch=default_branch,
            ac_branches=[
                (0, "ooo/test_ac_0"),
                (1, "ooo/test_ac_1"),
            ],
        )

        assert plan.all_succeeded

        # Original file untouched
        assert (repo / "config.yaml").read_text() == "version: 1\n"
        # Both AC-specific files present
        assert (repo / "notes_ac0.txt").read_text() == "AC 0 notes\n"
        assert (repo / "notes_ac1.txt").read_text() == "AC 1 notes\n"
        # Portfolio changes from both ACs present
        final = (repo / "portfolio.typ").read_text()
        assert "Built a real-time dashboard" in final
        assert "High-throughput API gateway" in final

    def test_rerun_produces_same_result(
        self, portfolio_repo: Path, default_branch: str
    ) -> None:
        """Re-running the same scenario twice produces identical results.

        This validates idempotency: the merge system produces deterministic
        output regardless of timing variations in parallel execution.
        """
        repo = portfolio_repo

        def run_scenario() -> str:
            """Run the full 3-AC merge scenario and return final content."""
            # Reset to clean state
            _git(["checkout", default_branch], repo)
            # Clean up any leftover branches
            for i in range(3):
                branch = f"ooo/rerun_ac_{i}"
                try:
                    _git(["branch", "-D", branch], repo)
                except subprocess.CalledProcessError:
                    pass

            # Reset to pre-merge state
            _git(["reset", "--hard", "HEAD"], repo)

            ac_contents = [AC0_CONTENT, AC1_CONTENT, AC2_CONTENT]
            for i, content in enumerate(ac_contents):
                branch = f"ooo/rerun_ac_{i}"
                _git(["checkout", "-b", branch, default_branch], repo)
                (repo / "portfolio.typ").write_text(content)
                _git(["add", "portfolio.typ"], repo)
                _git(["commit", "-m", f"AC {i} update"], repo)
                _git(["checkout", default_branch], repo)

            merger = WorktreeMerger(repo_root=repo)
            plan = merger.merge_all(
                target_branch=default_branch,
                ac_branches=[(i, f"ooo/rerun_ac_{i}") for i in range(3)],
            )
            assert plan.all_succeeded
            return (repo / "portfolio.typ").read_text()

        # Run 1
        result_1 = run_scenario()

        # Reset for re-run: go back to the commit before merges
        _git(["reset", "--hard", "HEAD~3"], repo)  # Undo 3 merge commits

        # Run 2
        result_2 = run_scenario()

        # Both runs should produce identical output
        assert result_1 == result_2

        # And all content should be present
        for expected in EXPECTED_STRINGS:
            assert expected in result_1

    def test_merge_order_determinism(
        self, portfolio_repo: Path, default_branch: str
    ) -> None:
        """Merge results are deterministic regardless of input order.

        WorktreeMerger.merge_all() sorts by ac_index, so passing
        branches in any order should produce the same final state.
        """
        repo = portfolio_repo

        # Create branches
        for i, content in enumerate([AC0_CONTENT, AC1_CONTENT, AC2_CONTENT]):
            _git(["checkout", "-b", f"ooo/order_ac_{i}", default_branch], repo)
            (repo / "portfolio.typ").write_text(content)
            _git(["add", "portfolio.typ"], repo)
            _git(["commit", "-m", f"AC {i}"], repo)
            _git(["checkout", default_branch], repo)

        # Merge in reverse order (2, 1, 0 input)
        merger = WorktreeMerger(repo_root=repo)
        plan = merger.merge_all(
            target_branch=default_branch,
            ac_branches=[
                (2, "ooo/order_ac_2"),
                (0, "ooo/order_ac_0"),
                (1, "ooo/order_ac_1"),
            ],
        )

        assert plan.all_succeeded
        # Verify sorted processing order
        assert plan.results[0].ac_index == 0
        assert plan.results[1].ac_index == 1
        assert plan.results[2].ac_index == 2

        # All changes present
        final = (repo / "portfolio.typ").read_text()
        for expected in EXPECTED_STRINGS:
            assert expected in final


class TestPortfolioOverlapPrediction:
    """Verify file overlap prediction identifies portfolio.typ as shared."""

    def test_prediction_detects_portfolio_overlap(self) -> None:
        """All 3 ACs mentioning portfolio.typ are predicted to overlap."""
        file_index = frozenset([
            "portfolio.typ",
            "README.md",
            "config.yaml",
            "src/main.py",
        ])

        specs = [
            ACDependencySpec(
                index=0,
                content="Update projects 1 & 2 in portfolio.typ with Web Dashboard and Mobile App descriptions",
                metadata={"files": ["portfolio.typ"]},
            ),
            ACDependencySpec(
                index=1,
                content="Update projects 3 & 4 in portfolio.typ with API Gateway and Data Pipeline descriptions",
                metadata={"files": ["portfolio.typ"]},
            ),
            ACDependencySpec(
                index=2,
                content="Update projects 5 & 6 in portfolio.typ with ML Service and CLI Tool descriptions",
                metadata={"files": ["portfolio.typ"]},
            ),
        ]

        prediction = asyncio.run(
            predict_file_overlaps(
                specs,
                file_index=file_index,
                stage_ac_indices=(0, 1, 2),
            )
        )

        # All 3 ACs should be flagged for isolation
        assert prediction.has_overlaps, "Expected overlaps but none detected"
        assert len(prediction.overlap_groups) >= 1

        # All ACs should be in the isolated set
        assert 0 in prediction.isolated_ac_indices
        assert 1 in prediction.isolated_ac_indices
        assert 2 in prediction.isolated_ac_indices
        assert len(prediction.shared_ac_indices) == 0

        # portfolio.typ should be in the shared paths of at least one group
        all_shared_paths: set[str] = set()
        for group in prediction.overlap_groups:
            all_shared_paths.update(group.shared_paths)
        assert "portfolio.typ" in all_shared_paths

    def test_prediction_with_non_overlapping_ac(self) -> None:
        """An AC not touching portfolio.typ stays in shared workspace."""
        file_index = frozenset([
            "portfolio.typ",
            "README.md",
            "tests/test_main.py",
        ])

        specs = [
            ACDependencySpec(
                index=0,
                content="Update portfolio.typ with project descriptions",
                metadata={"files": ["portfolio.typ"]},
            ),
            ACDependencySpec(
                index=1,
                content="Update portfolio.typ with more project descriptions",
                metadata={"files": ["portfolio.typ"]},
            ),
            ACDependencySpec(
                index=2,
                content="Add unit tests for the validation module",
                metadata={"files": ["tests/test_main.py"]},
            ),
        ]

        prediction = asyncio.run(
            predict_file_overlaps(
                specs,
                file_index=file_index,
                stage_ac_indices=(0, 1, 2),
            )
        )

        # ACs 0 and 1 should overlap (both touch portfolio.typ)
        assert 0 in prediction.isolated_ac_indices
        assert 1 in prediction.isolated_ac_indices
        # AC 2 should be safe in shared workspace
        assert 2 in prediction.shared_ac_indices

    def test_isolation_plan_from_prediction(self) -> None:
        """Isolation classification correctly separates overlapping ACs."""
        # Given overlap groups from prediction
        plan = classify_isolation_modes(
            ac_indices=[0, 1, 2],
            file_overlap_groups=[[0, 1, 2]],
        )

        # All 3 ACs should get WORKTREE mode
        assert needs_worktree(plan, 0) is True
        assert needs_worktree(plan, 1) is True
        assert needs_worktree(plan, 2) is True
        assert plan.all_shared is False
        assert plan.has_worktrees is True
        assert len(plan.worktree_indices) == 3


class TestPortfolioConflictDetection:
    """Test the conflict path for overlapping hunks in portfolio.typ."""

    def test_overlapping_hunks_detected_as_conflict(
        self, portfolio_repo: Path, default_branch: str
    ) -> None:
        """When two ACs modify the same project section, conflict is detected.

        This simulates the case where the section boundaries aren't
        large enough for git's diff algorithm to separate the hunks.
        """
        repo = portfolio_repo

        # Both ACs modify the same section (Project 1)
        ac0_overlapping = PORTFOLIO_TEMPLATE.replace(
            "Description: Placeholder for project 1.",
            "Description: Dashboard with React by AC 0.",
        )
        ac1_overlapping = PORTFOLIO_TEMPLATE.replace(
            "Description: Placeholder for project 1.",
            "Description: Dashboard with Vue by AC 1.",
        )

        _git(["checkout", "-b", "ooo/conflict_ac_0", default_branch], repo)
        (repo / "portfolio.typ").write_text(ac0_overlapping)
        _git(["add", "portfolio.typ"], repo)
        _git(["commit", "-m", "AC 0 updates project 1"], repo)
        _git(["checkout", default_branch], repo)

        _git(["checkout", "-b", "ooo/conflict_ac_1", default_branch], repo)
        (repo / "portfolio.typ").write_text(ac1_overlapping)
        _git(["add", "portfolio.typ"], repo)
        _git(["commit", "-m", "AC 1 updates project 1"], repo)
        _git(["checkout", default_branch], repo)

        merger = WorktreeMerger(repo_root=repo)

        # First merge succeeds
        r0 = merger.merge_ac_branch(default_branch, "ooo/conflict_ac_0", 0)
        assert r0.succeeded

        # Second merge should conflict (same line modified)
        r1 = merger.merge_ac_branch(default_branch, "ooo/conflict_ac_1", 1)
        assert r1.has_conflicts
        assert "portfolio.typ" in r1.conflicting_files

        # Repo should be clean (merge aborted)
        status = _git(["status", "--porcelain"], repo)
        assert status == ""

    def test_conflict_captures_diff_for_merge_agent(
        self, portfolio_repo: Path, default_branch: str
    ) -> None:
        """Conflict result includes diff text for merge-agent consumption."""
        repo = portfolio_repo

        # Same scenario: overlapping edits to project 1
        ac0 = PORTFOLIO_TEMPLATE.replace(
            "Status: TODO\nDescription: Placeholder for project 1.",
            "Status: Done\nDescription: AC 0 version.",
        )
        ac1 = PORTFOLIO_TEMPLATE.replace(
            "Status: TODO\nDescription: Placeholder for project 1.",
            "Status: Done\nDescription: AC 1 version.",
        )

        _git(["checkout", "-b", "ooo/diff_ac_0", default_branch], repo)
        (repo / "portfolio.typ").write_text(ac0)
        _git(["add", "portfolio.typ"], repo)
        _git(["commit", "-m", "AC 0"], repo)
        _git(["checkout", default_branch], repo)

        _git(["checkout", "-b", "ooo/diff_ac_1", default_branch], repo)
        (repo / "portfolio.typ").write_text(ac1)
        _git(["add", "portfolio.typ"], repo)
        _git(["commit", "-m", "AC 1"], repo)
        _git(["checkout", default_branch], repo)

        merger = WorktreeMerger(repo_root=repo)
        merger.merge_ac_branch(default_branch, "ooo/diff_ac_0", 0)
        r1 = merger.merge_ac_branch(default_branch, "ooo/diff_ac_1", 1)

        assert r1.has_conflicts
        # conflict_diff should contain the diff text for merge-agent
        assert r1.conflict_diff  # non-empty
        assert "portfolio.typ" in r1.conflicting_files


class TestPortfolioEndToEndPipeline:
    """Full pipeline: predict → isolate → execute → merge → verify."""

    def test_full_pipeline_six_projects_three_acs(
        self, portfolio_repo: Path, default_branch: str
    ) -> None:
        """Complete end-to-end pipeline for the portfolio.typ scenario.

        Simulates:
        1. DependencyAnalyzer predicts file overlap at planning time
        2. ACIsolationPlan classifies ACs for worktree isolation
        3. ACWorktreeManager creates per-AC worktrees
        4. Each AC modifies portfolio.typ in its worktree
        5. Changes are committed in each worktree
        6. WorktreeMerger merges all branches back
        7. Final file contains all 6 project updates
        """
        repo = portfolio_repo
        execution_id = "orch_portfolio_test"

        # Step 1: Simulate file overlap prediction
        file_index = frozenset(["portfolio.typ", "README.md"])
        specs = [
            ACDependencySpec(
                index=i,
                content=f"Update portfolio.typ projects {i*2+1} & {i*2+2}",
                metadata={"files": ["portfolio.typ"]},
            )
            for i in range(3)
        ]

        prediction = asyncio.run(
            predict_file_overlaps(specs, file_index=file_index)
        )
        assert prediction.has_overlaps

        # Step 2: Classify isolation modes
        plan = classify_isolation_modes(
            ac_indices=[0, 1, 2],
            file_overlap_groups=[
                list(g.ac_indices) for g in prediction.overlap_groups
            ],
        )
        assert plan.has_worktrees
        for i in range(3):
            assert needs_worktree(plan, i)

        # Step 3: Create per-AC worktrees
        manager = ACWorktreeManager(
            execution_id=execution_id,
            repo_root=str(repo),
            source_cwd=str(repo),
        )

        worktrees: dict[int, ACWorktreeInfo] = {}
        for i in range(3):
            info = manager.create_ac_worktree(i)
            worktrees[i] = info
            assert Path(info.worktree_path).exists()
            assert info.branch == f"ooo/{execution_id}_ac_{i}"

        # Step 4: Each AC modifies portfolio.typ in its own worktree
        ac_contents = [AC0_CONTENT, AC1_CONTENT, AC2_CONTENT]
        for i, content in enumerate(ac_contents):
            wt_path = Path(worktrees[i].worktree_path)
            (wt_path / "portfolio.typ").write_text(content)

        # Step 5: Commit changes in each worktree
        for i in range(3):
            sha = manager.commit_ac_changes(i, f"AC {i}: Update portfolio projects")
            assert sha is not None, f"AC {i} should have changes to commit"

        # Step 6: Merge all branches back to the default branch
        merger = WorktreeMerger(repo_root=repo)
        merge_plan = merger.merge_all(
            target_branch=default_branch,
            ac_branches=[
                (i, worktrees[i].branch) for i in range(3)
            ],
        )

        assert merge_plan.all_succeeded, (
            f"Merge failed: {[(r.ac_index, r.outcome.value) for r in merge_plan.results]}"
        )
        assert merge_plan.success_count == 3

        # Step 7: Verify ALL 6 project descriptions are present
        # Switch back to default branch to read the final state
        _git(["checkout", default_branch], repo)
        final_content = (repo / "portfolio.typ").read_text()

        for expected in EXPECTED_STRINGS:
            assert expected in final_content, (
                f"DATA LOSS: Missing AC change '{expected}' in final portfolio.typ"
            )

        # Verify all 6 projects are marked complete
        assert final_content.count("Status: Complete") == 6, (
            f"Expected 6 'Status: Complete' but found "
            f"{final_content.count('Status: Complete')}"
        )
        assert "Placeholder for project" not in final_content

        # Cleanup worktrees
        removed = manager.remove_all(force=True)
        assert removed == 3

    def test_full_pipeline_repeated_execution_no_data_loss(
        self, portfolio_repo: Path, default_branch: str
    ) -> None:
        """Re-running the full pipeline produces the same result.

        This validates AC 7's core requirement: re-running the portfolio.typ
        scenario results in all ACs' changes present with no data loss.
        """
        repo = portfolio_repo

        def run_full_pipeline(run_id: str) -> str:
            """Execute the complete pipeline and return final portfolio content."""
            execution_id = f"orch_portfolio_{run_id}"

            # Create worktrees
            manager = ACWorktreeManager(
                execution_id=execution_id,
                repo_root=str(repo),
                source_cwd=str(repo),
            )

            worktrees: dict[int, ACWorktreeInfo] = {}
            for i in range(3):
                worktrees[i] = manager.create_ac_worktree(i)

            # Write AC changes in worktrees
            ac_contents = [AC0_CONTENT, AC1_CONTENT, AC2_CONTENT]
            for i, content in enumerate(ac_contents):
                wt_path = Path(worktrees[i].worktree_path)
                (wt_path / "portfolio.typ").write_text(content)

            # Commit in each worktree
            for i in range(3):
                manager.commit_ac_changes(i, f"AC {i} update ({run_id})")

            # Merge all branches
            merger = WorktreeMerger(repo_root=repo)
            merge_plan = merger.merge_all(
                target_branch=default_branch,
                ac_branches=[(i, worktrees[i].branch) for i in range(3)],
            )
            assert merge_plan.all_succeeded

            # Read result
            _git(["checkout", default_branch], repo)
            result = (repo / "portfolio.typ").read_text()

            # Cleanup
            manager.remove_all(force=True)

            return result

        # Run 1
        result_1 = run_full_pipeline("run1")

        # Reset repo to pre-merge state for run 2
        _git(["reset", "--hard", "HEAD~3"], repo)

        # Run 2
        result_2 = run_full_pipeline("run2")

        # Both runs should contain all content
        for expected in EXPECTED_STRINGS:
            assert expected in result_1, f"Run 1 missing: {expected}"
            assert expected in result_2, f"Run 2 missing: {expected}"

        # Content should be identical
        assert result_1 == result_2, "Re-run produced different content — non-deterministic!"


class TestPortfolioWorktreeLifecycle:
    """Verify worktree lifecycle specific to the portfolio scenario."""

    def test_worktree_isolation_prevents_cross_ac_interference(
        self, portfolio_repo: Path, default_branch: str
    ) -> None:
        """Changes in one AC's worktree don't appear in another's."""
        repo = portfolio_repo
        manager = ACWorktreeManager(
            execution_id="orch_isolation_test",
            repo_root=str(repo),
            source_cwd=str(repo),
        )

        wt0 = manager.create_ac_worktree(0)
        wt1 = manager.create_ac_worktree(1)

        # AC 0 writes its version
        (Path(wt0.worktree_path) / "portfolio.typ").write_text(AC0_CONTENT)

        # AC 1's copy should still have the original template
        wt1_content = (Path(wt1.worktree_path) / "portfolio.typ").read_text()
        assert wt1_content == PORTFOLIO_TEMPLATE
        assert "Built a real-time dashboard" not in wt1_content

        # AC 1 writes its own version
        (Path(wt1.worktree_path) / "portfolio.typ").write_text(AC1_CONTENT)

        # AC 0's copy should still have AC 0's version
        wt0_content = (Path(wt0.worktree_path) / "portfolio.typ").read_text()
        assert "Built a real-time dashboard" in wt0_content
        assert "High-throughput API gateway" not in wt0_content

        # Main repo should still have the template
        main_content = (repo / "portfolio.typ").read_text()
        assert main_content == PORTFOLIO_TEMPLATE

        manager.remove_all(force=True)

    def test_worktree_commit_captures_all_changes(
        self, portfolio_repo: Path, default_branch: str
    ) -> None:
        """commit_ac_changes() captures the full file content in each worktree."""
        repo = portfolio_repo
        manager = ACWorktreeManager(
            execution_id="orch_commit_test",
            repo_root=str(repo),
            source_cwd=str(repo),
        )

        wt = manager.create_ac_worktree(0)
        wt_path = Path(wt.worktree_path)

        # Write changes
        (wt_path / "portfolio.typ").write_text(AC0_CONTENT)

        # Commit
        sha = manager.commit_ac_changes(0, "AC 0 update")
        assert sha is not None

        # Verify the commit contains the changes
        diff_output = _git(
            ["diff", f"{default_branch}..{wt.branch}", "--", "portfolio.typ"],
            repo,
        )
        assert "Built a real-time dashboard" in diff_output

        manager.remove_all(force=True)

    def test_no_commit_when_no_changes(
        self, portfolio_repo: Path, default_branch: str
    ) -> None:
        """commit_ac_changes() returns None when there are no modifications."""
        repo = portfolio_repo
        manager = ACWorktreeManager(
            execution_id="orch_nochange_test",
            repo_root=str(repo),
            source_cwd=str(repo),
        )

        manager.create_ac_worktree(0)
        # Don't modify anything
        sha = manager.commit_ac_changes(0, "Should be empty")
        assert sha is None

        manager.remove_all(force=True)
