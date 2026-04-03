"""Conservative file overlap prediction for parallel AC isolation decisions.

Predicts which files each AC will touch before execution, using heuristics
like keyword-to-path mapping, import graph analysis, and broad category
expansion. The predictor is intentionally conservative — over-prediction
is preferred to under-prediction to ensure zero false negatives.

When predicted file sets overlap between ACs in the same execution stage,
those ACs are candidates for worktree isolation.

Usage:
    predictor = FileOverlapPredictor(repo_root="/path/to/repo")
    prediction = await predictor.predict(ac_specs, seed=seed)

    for group in prediction.overlap_groups:
        # ACs in this group share predicted file paths
        ...
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import TYPE_CHECKING

from ouroboros.observability.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ouroboros.core.seed import BrownfieldContext, Seed
    from ouroboros.orchestrator.dependency_analyzer import ACDependencySpec

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ACFilePrediction:
    """Predicted file footprint for a single AC.

    Attributes:
        ac_index: Zero-based AC index.
        predicted_paths: Conservatively predicted file/directory paths (relative
            to repo root). Includes direct matches and expanded neighbours.
        categories: Semantic categories matched by the AC description.
        confidence: 0.0–1.0 estimate of prediction quality.
    """

    ac_index: int
    predicted_paths: frozenset[str] = field(default_factory=frozenset)
    categories: frozenset[str] = field(default_factory=frozenset)
    confidence: float = 0.5


@dataclass(frozen=True, slots=True)
class OverlapGroup:
    """A group of ACs whose predicted file sets overlap.

    Attributes:
        ac_indices: ACs that share at least one predicted path.
        shared_paths: Paths predicted by two or more ACs in this group.
    """

    ac_indices: tuple[int, ...]
    shared_paths: frozenset[str]


@dataclass(frozen=True, slots=True)
class FileOverlapPrediction:
    """Result of file overlap prediction across a set of ACs.

    Attributes:
        ac_predictions: Per-AC predicted file footprints.
        overlap_groups: Groups of ACs with overlapping file predictions.
        isolated_ac_indices: ACs that need worktree isolation.
        shared_ac_indices: ACs safe to run in shared workspace.
    """

    ac_predictions: tuple[ACFilePrediction, ...]
    overlap_groups: tuple[OverlapGroup, ...] = field(default_factory=tuple)
    isolated_ac_indices: frozenset[int] = field(default_factory=frozenset)
    shared_ac_indices: frozenset[int] = field(default_factory=frozenset)

    @property
    def has_overlaps(self) -> bool:
        """True when at least one overlap group was detected."""
        return len(self.overlap_groups) > 0

    @property
    def all_ac_indices(self) -> frozenset[int]:
        """Return all AC indices covered by this prediction."""
        return frozenset(p.ac_index for p in self.ac_predictions)


# ---------------------------------------------------------------------------
# Keyword → category → path mapping tables
# ---------------------------------------------------------------------------

# Semantic categories that map keywords in AC text to broad file areas.
# Each category expands to directory prefixes/globs that are matched against
# the repository file index. Conservative: a single keyword triggers an
# entire category.

_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "test": (
        "test", "tests", "testing", "spec", "specs", "unittest",
        "pytest", "coverage", "fixture", "mock", "assert",
    ),
    "config": (
        "config", "configuration", "settings", "env", "environment",
        "pyproject", "setup.cfg", "setup.py", "toml", "yaml", "yml",
        "ini", ".env",
    ),
    "docs": (
        "doc", "docs", "documentation", "readme", "changelog",
        "contributing", "license", "guide",
    ),
    "ci": (
        "ci", "cd", "pipeline", "workflow", "github actions",
        "circleci", "jenkins", "deploy", "deployment",
    ),
    "model": (
        "model", "models", "schema", "entity", "entities",
        "dataclass", "pydantic", "orm", "migration",
    ),
    "api": (
        "api", "endpoint", "route", "routes", "handler", "handlers",
        "controller", "controllers", "view", "views", "rest", "graphql",
    ),
    "cli": (
        "cli", "command", "commands", "argparse", "click", "typer",
        "subcommand",
    ),
    "core": (
        "core", "base", "foundation", "types", "errors", "exceptions",
        "utils", "utilities", "helpers", "common", "shared",
    ),
    "orchestrator": (
        "orchestrator", "orchestration", "executor", "execution",
        "parallel", "coordinator", "dependency", "stage", "level",
        "runtime", "adapter", "session", "workflow",
    ),
    "seed": (
        "seed", "specification", "acceptance criteria", "ontology",
        "evaluation", "interview",
    ),
    "ui": (
        "ui", "tui", "frontend", "component", "widget", "template",
        "render", "display", "screen",
    ),
    "provider": (
        "provider", "llm", "anthropic", "openai", "claude",
        "completion", "chat",
    ),
    "observability": (
        "log", "logging", "logger", "metric", "metrics", "trace",
        "tracing", "telemetry", "observability", "monitoring",
    ),
    "plugin": (
        "plugin", "plugins", "extension", "extensions", "hook",
        "hooks", "middleware", "skill", "skills", "agent", "agents",
    ),
}

# Category → common directory prefixes (relative to repo root).
# These are expanded with actual filesystem discovery at prediction time.
_CATEGORY_DIRECTORY_HINTS: dict[str, tuple[str, ...]] = {
    "test": ("tests/", "test/", "spec/"),
    "config": ("", "config/", "configs/"),
    "docs": ("docs/", "doc/"),
    "ci": (".github/", ".circleci/", ".gitlab-ci/"),
    "model": ("src/", "models/", "schemas/"),
    "api": ("src/", "api/"),
    "cli": ("src/", "commands/", "cli/"),
    "core": ("src/", "core/", "lib/"),
    "orchestrator": ("src/ouroboros/orchestrator/",),
    "seed": ("src/ouroboros/core/", "src/ouroboros/"),
    "ui": ("src/", "ui/", "tui/"),
    "provider": ("src/ouroboros/providers/",),
    "observability": ("src/ouroboros/observability/",),
    "plugin": ("plugins/", "skills/", "hooks/", "agents/", "src/ouroboros/agents/"),
}


# Filename patterns that match common project files
_COMMON_SHARED_FILES: tuple[str, ...] = (
    "__init__.py",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "Makefile",
    "CHANGELOG.md",
    "README.md",
)


# ---------------------------------------------------------------------------
# Path extraction patterns
# ---------------------------------------------------------------------------

# Match explicit file paths in AC text (e.g., "src/foo/bar.py", "./config.yaml")
_PATH_PATTERN = re.compile(
    r"""(?:^|[\s"'`(,])"""  # boundary
    r"""("""
    r"""(?:\.{0,2}/)?"""  # optional ./ or ../
    r"""(?:[a-zA-Z0-9_\-]+/)"""  # at least one directory component
    r"""[a-zA-Z0-9_\-]+"""  # filename stem
    r"""(?:\.[a-zA-Z0-9]+)?"""  # optional extension
    r""")"""
    r"""(?:[\s"'`),.:;]|$)""",  # boundary
    re.MULTILINE,
)

# Match module-style references (e.g., "ouroboros.orchestrator.coordinator")
_MODULE_PATTERN = re.compile(
    r"""(?:^|[\s"'`(,])"""
    r"""((?:[a-zA-Z_][a-zA-Z0-9_]*\.){2,}[a-zA-Z_][a-zA-Z0-9_]*)"""
    r"""(?:[\s"'`),.:;]|$)""",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Predictor implementation
# ---------------------------------------------------------------------------


class FileOverlapPredictor:
    """Conservatively predicts file overlap between parallel ACs.

    The predictor uses multiple heuristics and unions their results to
    ensure conservative (over-predicting) behaviour:

    1. **Explicit path extraction** — file paths and module references
       mentioned directly in AC text.
    2. **Keyword-to-category mapping** — broad semantic categories
       triggered by keywords in the AC description.
    3. **Category-to-directory expansion** — each category maps to
       repository directory prefixes; all files under those prefixes
       are included.
    4. **Brownfield context integration** — uses seed's brownfield
       context_references to anchor predictions to actual codebase
       structure.
    5. **Neighbour expansion** — for each predicted file, sibling files
       in the same directory are included as potential collateral.

    All heuristics are unioned (not intersected), so any single signal
    is sufficient to predict a file — guaranteeing zero false negatives
    at the cost of some false positives.
    """

    def __init__(
        self,
        repo_root: str | Path | None = None,
        *,
        file_index: frozenset[str] | None = None,
    ) -> None:
        """Initialize the predictor.

        Args:
            repo_root: Repository root for filesystem discovery. If None,
                predictions rely solely on text-based heuristics.
            file_index: Pre-built set of known repo-relative file paths.
                If None and repo_root is provided, discovered lazily.
        """
        self._repo_root = Path(repo_root) if repo_root else None
        self._file_index = file_index
        self._dir_index: frozenset[str] | None = None

    @property
    def file_index(self) -> frozenset[str]:
        """Lazily discover and cache the repository file index."""
        if self._file_index is None:
            self._file_index = self._discover_file_index()
        return self._file_index

    @property
    def dir_index(self) -> frozenset[str]:
        """Lazily discover and cache the repository directory index."""
        if self._dir_index is None:
            dirs: set[str] = set()
            for path in self.file_index:
                parts = Path(path).parts
                for i in range(1, len(parts)):
                    dirs.add(str(Path(*parts[:i])) + "/")
            self._dir_index = frozenset(dirs)
        return self._dir_index

    def _discover_file_index(self) -> frozenset[str]:
        """Build the file index from the repository root."""
        if self._repo_root is None:
            return frozenset()

        paths: set[str] = set()
        try:
            import subprocess

            result = subprocess.run(
                ["git", "ls-files"],
                cwd=self._repo_root,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if line:
                        paths.add(line)
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            log.warning("file_overlap_predictor.git_ls_files_failed")

        return frozenset(paths)

    async def predict(
        self,
        ac_specs: Sequence[ACDependencySpec],
        *,
        seed: Seed | None = None,
        stage_ac_indices: tuple[int, ...] | None = None,
    ) -> FileOverlapPrediction:
        """Predict file overlaps for ACs in a parallel execution stage.

        Args:
            ac_specs: AC specifications to analyse.
            seed: Optional seed for brownfield context integration.
            stage_ac_indices: If provided, only predict for these AC indices.
                ACs not in this set are ignored for overlap computation.

        Returns:
            FileOverlapPrediction with per-AC footprints and overlap groups.
        """
        target_indices = frozenset(stage_ac_indices) if stage_ac_indices else None
        brownfield = seed.brownfield_context if seed else None

        predictions: list[ACFilePrediction] = []
        for spec in ac_specs:
            if target_indices is not None and spec.index not in target_indices:
                continue
            prediction = self._predict_single_ac(spec, brownfield)
            predictions.append(prediction)

        overlap_groups = self._compute_overlap_groups(predictions)
        isolated, shared = self._partition_acs(predictions, overlap_groups)

        result = FileOverlapPrediction(
            ac_predictions=tuple(predictions),
            overlap_groups=tuple(overlap_groups),
            isolated_ac_indices=isolated,
            shared_ac_indices=shared,
        )

        log.info(
            "file_overlap_predictor.prediction_complete",
            ac_count=len(predictions),
            overlap_groups=len(overlap_groups),
            isolated_count=len(isolated),
            shared_count=len(shared),
        )

        return result

    def _predict_single_ac(
        self,
        spec: ACDependencySpec,
        brownfield: BrownfieldContext | None,
    ) -> ACFilePrediction:
        """Predict file footprint for a single AC."""
        text = self._build_analysis_text(spec)

        # Heuristic 1: Extract explicit file paths from AC text
        explicit_paths = self._extract_explicit_paths(text)

        # Heuristic 2: Extract module references and convert to paths
        module_paths = self._extract_module_paths(text)

        # Heuristic 3: Keyword → category matching
        categories = self._match_categories(text)

        # Heuristic 4: Category → directory expansion
        category_paths = self._expand_categories(categories)

        # Heuristic 5: Brownfield context integration
        brownfield_paths = self._integrate_brownfield(text, brownfield)

        # Heuristic 6: Metadata-driven path hints
        metadata_paths = self._extract_metadata_paths(spec)

        # Union all predicted paths (conservative)
        all_paths = (
            explicit_paths
            | module_paths
            | category_paths
            | brownfield_paths
            | metadata_paths
        )

        # Heuristic 7: Neighbour expansion — add siblings for each predicted file
        expanded = self._expand_neighbours(all_paths)

        # Resolve against file index for grounding
        grounded = self._ground_to_file_index(expanded)

        # Confidence: higher when we have explicit paths, lower when relying on categories
        confidence = self._estimate_confidence(
            explicit_paths, module_paths, metadata_paths, categories, grounded,
        )

        return ACFilePrediction(
            ac_index=spec.index,
            predicted_paths=grounded,
            categories=categories,
            confidence=confidence,
        )

    def _build_analysis_text(self, spec: ACDependencySpec) -> str:
        """Build combined text for analysis from AC spec fields."""
        parts = [spec.content]
        for key in ("description", "context", "notes", "details"):
            val = spec.metadata.get(key)
            if isinstance(val, str) and val.strip():
                parts.append(val)
            val = spec.context.get(key)
            if isinstance(val, str) and val.strip():
                parts.append(val)
        # Include file hints from metadata
        for key in ("files", "file_paths", "touches", "modifies"):
            val = spec.metadata.get(key)
            if isinstance(val, (list, tuple)):
                parts.extend(str(v) for v in val)
            elif isinstance(val, str):
                parts.append(val)
        return "\n".join(parts).lower()

    def _extract_explicit_paths(self, text: str) -> frozenset[str]:
        """Extract explicit file paths from text."""
        paths: set[str] = set()
        for match in _PATH_PATTERN.finditer(text):
            raw = match.group(1).strip().strip("'\"`,")
            # Normalize: remove leading ./ or ../
            normalized = re.sub(r"^\.{1,2}/", "", raw)
            if normalized:
                paths.add(normalized)
        return frozenset(paths)

    def _extract_module_paths(self, text: str) -> frozenset[str]:
        """Convert dotted module references to file paths."""
        paths: set[str] = set()
        for match in _MODULE_PATTERN.finditer(text):
            module = match.group(1).strip()
            # Convert dots to slashes: ouroboros.core.types → ouroboros/core/types
            file_path = module.replace(".", "/")
            # Add both as directory and as .py file
            paths.add(file_path + ".py")
            paths.add(file_path + "/")
        return frozenset(paths)

    def _match_categories(self, text: str) -> frozenset[str]:
        """Match AC text against category keyword tables."""
        matched: set[str] = set()
        # Tokenize text for word-boundary matching
        words = set(re.findall(r"[a-z][a-z0-9_]+", text))

        for category, keywords in _CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                # Check for multi-word keywords via substring
                if " " in keyword:
                    if keyword in text:
                        matched.add(category)
                        break
                elif keyword in words:
                    matched.add(category)
                    break

        return frozenset(matched)

    def _expand_categories(self, categories: frozenset[str]) -> frozenset[str]:
        """Expand matched categories to directory-anchored path prefixes."""
        paths: set[str] = set()
        for category in categories:
            hints = _CATEGORY_DIRECTORY_HINTS.get(category, ())
            for hint in hints:
                if not hint:
                    # Root-level config files
                    for f in _COMMON_SHARED_FILES:
                        paths.add(f)
                    continue
                # Add the directory hint itself as a prefix
                paths.add(hint)
                # Match files from the index under this prefix
                for indexed_path in self.file_index:
                    if indexed_path.startswith(hint):
                        paths.add(indexed_path)
        return frozenset(paths)

    def _integrate_brownfield(
        self,
        text: str,
        brownfield: BrownfieldContext | None,
    ) -> frozenset[str]:
        """Use brownfield context to anchor predictions to real paths."""
        if brownfield is None or brownfield.project_type == "greenfield":
            return frozenset()

        paths: set[str] = set()
        for ref in brownfield.context_references:
            if ref.role != "primary":
                continue
            # If the AC text mentions any part of this reference path,
            # conservatively include the entire referenced directory
            ref_path = ref.path.rstrip("/")
            ref_name = Path(ref_path).name.lower()
            ref_parts = {p.lower() for p in Path(ref_path).parts}

            # Check if AC mentions the referenced module name
            if ref_name in text or any(part in text for part in ref_parts if len(part) > 3):
                # Make relative to repo root if possible
                if self._repo_root:
                    try:
                        rel = str(Path(ref_path).relative_to(self._repo_root))
                        paths.add(rel + "/")
                        # Include all indexed files under this directory
                        for indexed_path in self.file_index:
                            if indexed_path.startswith(rel + "/") or indexed_path.startswith(rel):
                                paths.add(indexed_path)
                    except ValueError:
                        paths.add(ref_path + "/")
                else:
                    paths.add(ref_path + "/")

        return frozenset(paths)

    def _extract_metadata_paths(self, spec: ACDependencySpec) -> frozenset[str]:
        """Extract file path hints from AC metadata and context."""
        paths: set[str] = set()
        for source in (spec.metadata, spec.context):
            for key in ("files", "file_paths", "touches", "modifies", "creates", "reads"):
                raw = source.get(key)
                if raw is None:
                    continue
                if isinstance(raw, str):
                    paths.add(raw.strip())
                elif isinstance(raw, (list, tuple)):
                    for item in raw:
                        if isinstance(item, str) and item.strip():
                            paths.add(item.strip())
        return frozenset(paths)

    def _expand_neighbours(self, paths: frozenset[str]) -> frozenset[str]:
        """Add sibling files in the same directory for each predicted path.

        This is a conservative expansion — if we predict a file, we also
        predict its neighbours since ACs working on one file in a directory
        frequently touch others.
        """
        expanded: set[str] = set(paths)
        seen_dirs: set[str] = set()

        for path in paths:
            if path.endswith("/"):
                # Already a directory — include all files underneath
                for indexed_path in self.file_index:
                    if indexed_path.startswith(path):
                        expanded.add(indexed_path)
                continue

            parent = str(Path(path).parent)
            if parent == "." or parent in seen_dirs:
                continue
            seen_dirs.add(parent)

            parent_prefix = parent + "/"
            for indexed_path in self.file_index:
                if indexed_path.startswith(parent_prefix):
                    # Only include direct children, not deep descendants
                    remainder = indexed_path[len(parent_prefix):]
                    if "/" not in remainder:
                        expanded.add(indexed_path)

        return frozenset(expanded)

    def _ground_to_file_index(self, paths: frozenset[str]) -> frozenset[str]:
        """Intersect predicted paths with the file index for grounding.

        Paths that look like directories expand to all indexed files
        underneath. Paths not found in the index are kept as-is
        (they may represent files that will be created).
        """
        if not self.file_index:
            return paths

        grounded: set[str] = set()
        for path in paths:
            if path in self.file_index:
                grounded.add(path)
            elif path.endswith("/"):
                # Directory prefix — expand to all files underneath
                for indexed_path in self.file_index:
                    if indexed_path.startswith(path):
                        grounded.add(indexed_path)
            else:
                # Check if it's a prefix of any indexed path
                prefix = path.rstrip("/") + "/"
                matched = False
                for indexed_path in self.file_index:
                    if indexed_path.startswith(prefix):
                        grounded.add(indexed_path)
                        matched = True
                if not matched:
                    # Keep unmatched paths — they may be new files
                    grounded.add(path)

        return frozenset(grounded)

    def _estimate_confidence(
        self,
        explicit_paths: frozenset[str],
        module_paths: frozenset[str],
        metadata_paths: frozenset[str],
        categories: frozenset[str],
        grounded: frozenset[str],
    ) -> float:
        """Estimate prediction confidence based on signal quality."""
        if not grounded:
            return 0.1

        # More explicit signals → higher confidence
        score = 0.3  # baseline
        if explicit_paths:
            score += 0.25
        if module_paths:
            score += 0.15
        if metadata_paths:
            score += 0.25
        if categories and len(categories) <= 3:
            score += 0.1
        # Penalize very broad predictions
        if len(grounded) > 100:
            score -= 0.1
        if len(categories) > 5:
            score -= 0.1

        return max(0.1, min(1.0, score))

    def _compute_overlap_groups(
        self,
        predictions: list[ACFilePrediction],
    ) -> list[OverlapGroup]:
        """Identify groups of ACs with overlapping predicted file sets.

        Uses union-find to merge ACs that share any predicted path into
        connected components.
        """
        if len(predictions) < 2:
            return []

        # Build path → AC indices mapping
        path_to_acs: dict[str, set[int]] = defaultdict(set)
        for pred in predictions:
            for path in pred.predicted_paths:
                path_to_acs[path].add(pred.ac_index)

        # Find paths shared by 2+ ACs
        shared_paths_per_pair: dict[frozenset[int], set[str]] = defaultdict(set)
        for path, ac_indices in path_to_acs.items():
            if len(ac_indices) >= 2:
                shared_paths_per_pair[frozenset(ac_indices)].add(path)

        if not shared_paths_per_pair:
            return []

        # Union-find to merge overlapping groups
        parent: dict[int, int] = {}

        def find(x: int) -> int:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for ac_set in shared_paths_per_pair:
            indices = sorted(ac_set)
            for i in range(1, len(indices)):
                union(indices[0], indices[i])

        # Collect groups
        groups: dict[int, set[int]] = defaultdict(set)
        all_overlapping = set()
        for ac_set in shared_paths_per_pair:
            all_overlapping.update(ac_set)
        for idx in all_overlapping:
            groups[find(idx)].add(idx)

        # Build OverlapGroup results
        overlap_groups: list[OverlapGroup] = []
        for _root, members in sorted(groups.items()):
            # Collect all shared paths for this group
            group_shared: set[str] = set()
            for path, ac_indices in path_to_acs.items():
                if len(ac_indices & members) >= 2:
                    group_shared.add(path)

            overlap_groups.append(
                OverlapGroup(
                    ac_indices=tuple(sorted(members)),
                    shared_paths=frozenset(group_shared),
                )
            )

        return overlap_groups

    def _partition_acs(
        self,
        predictions: list[ACFilePrediction],
        overlap_groups: list[OverlapGroup],
    ) -> tuple[frozenset[int], frozenset[int]]:
        """Partition ACs into isolated (need worktree) and shared (safe) sets.

        All ACs in any overlap group are marked for isolation.
        """
        isolated: set[int] = set()
        for group in overlap_groups:
            isolated.update(group.ac_indices)

        all_indices = {p.ac_index for p in predictions}
        shared = all_indices - isolated

        return frozenset(isolated), frozenset(shared)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


async def predict_file_overlaps(
    ac_specs: Sequence[ACDependencySpec],
    *,
    repo_root: str | Path | None = None,
    seed: Seed | None = None,
    stage_ac_indices: tuple[int, ...] | None = None,
    file_index: frozenset[str] | None = None,
) -> FileOverlapPrediction:
    """Convenience function to predict file overlaps for a set of ACs.

    Args:
        ac_specs: AC specifications to analyse.
        repo_root: Repository root for filesystem discovery.
        seed: Optional seed for brownfield context integration.
        stage_ac_indices: If provided, only predict for these AC indices.
        file_index: Pre-built set of known repo-relative file paths.

    Returns:
        FileOverlapPrediction with per-AC footprints and overlap groups.
    """
    predictor = FileOverlapPredictor(repo_root=repo_root, file_index=file_index)
    return await predictor.predict(
        ac_specs,
        seed=seed,
        stage_ac_indices=stage_ac_indices,
    )


__all__ = [
    "ACFilePrediction",
    "FileOverlapPrediction",
    "FileOverlapPredictor",
    "OverlapGroup",
    "predict_file_overlaps",
]
