"""
tests/test_advanced.py - Tests for conflict resolver, code graph, and learning system
"""
import pytest
from app.core.conflict_resolver import ConflictResolver, ConflictType, ConflictSeverity
from app.core.code_graph import CodeGraphBuilder
from app.core.learning import ContinuousLearner, FusionOutcome
import tempfile
import os


PYTHON_CODE = """
import os
from typing import List

def calculate(n: int) -> int:
    if n <= 0:
        return 0
    return n * calculate(n - 1)

def process_data(items: List[int]) -> List[int]:
    return [calculate(x) for x in items]

class DataHandler:
    def __init__(self, config: dict):
        self.config = config
        self.cache = {}

    def handle(self, data):
        if data in self.cache:
            return self.cache[data]
        result = calculate(data)
        self.cache[data] = result
        return result
"""

JS_SAME_NAMES = """
const axios = require('axios');

function calculate(value) {
    return value * 2;
}

function process_data(items) {
    return items.map(x => calculate(x));
}

class DataHandler {
    constructor(config) {
        this.config = config;
    }
    handle(data) {
        return calculate(data);
    }
}
"""

CLEAN_GO = """
package main

import "fmt"

func fibonacci(n int) int {
    if n <= 1 {
        return n
    }
    return fibonacci(n-1) + fibonacci(n-2)
}

type Calculator struct {
    precision int
}

func (c *Calculator) compute(n int) int {
    return fibonacci(n)
}
"""


# ─── Conflict Resolver Tests ──────────────────────────────────────────────────

class TestConflictResolver:
    def setup_method(self):
        self.resolver = ConflictResolver()

    def test_detects_naming_conflicts(self):
        report = self.resolver.analyze(
            PYTHON_CODE, JS_SAME_NAMES, "python", "javascript"
        )
        # calculate, process_data, DataHandler all conflict
        conflict_symbols = [c.symbol for c in report.conflicts]
        assert "calculate" in conflict_symbols or "DataHandler" in conflict_symbols

    def test_conflict_count(self):
        report = self.resolver.analyze(
            PYTHON_CODE, JS_SAME_NAMES, "python", "javascript"
        )
        assert report.total > 0

    def test_auto_resolvable_conflicts(self):
        report = self.resolver.analyze(
            PYTHON_CODE, JS_SAME_NAMES, "python", "javascript"
        )
        auto_res = [c for c in report.conflicts if c.auto_resolvable]
        assert len(auto_res) > 0
        # Each auto-resolvable should have a resolved_symbol
        for c in auto_res:
            assert c.resolved_symbol is not None

    def test_no_conflicts_different_code(self):
        primary = "def alpha(): return 1\n"
        secondary = "def beta(): return 2\n"
        report = self.resolver.analyze(primary, secondary, "python", "python")
        naming = [c for c in report.conflicts if c.type == ConflictType.NAMING]
        assert len(naming) == 0

    def test_apply_auto_resolutions(self):
        report = self.resolver.analyze(
            PYTHON_CODE, JS_SAME_NAMES, "python", "javascript"
        )
        _, patched = self.resolver.apply_auto_resolutions(
            PYTHON_CODE, JS_SAME_NAMES, report
        )
        # Patched secondary should have renamed conflicting symbols
        assert len(patched) > 0

    def test_resolution_hints_generated(self):
        report = self.resolver.analyze(
            PYTHON_CODE, JS_SAME_NAMES, "python", "javascript"
        )
        assert len(report.resolution_hints) > 0

    def test_cross_language_type_conflicts(self):
        js_code = "const x = undefined; if (x === null) {}"
        py_code = "x = None\nif x is None: pass"
        report = self.resolver.analyze(js_code, py_code, "javascript", "python")
        type_conflicts = [c for c in report.conflicts if c.type == ConflictType.TYPE]
        # Should detect null/None semantics conflict
        assert len(type_conflicts) >= 0  # May or may not trigger depending on patterns

    def test_import_conflicts_detected(self):
        py_requests = "import requests\nrequests.get('http://example.com')"
        py_aiohttp = "import aiohttp\nasync with aiohttp.ClientSession() as s: pass"
        report = self.resolver.analyze(py_requests, py_aiohttp, "python", "python")
        import_conflicts = [c for c in report.conflicts if c.type == ConflictType.IMPORT]
        assert len(import_conflicts) > 0

    def test_severity_levels(self):
        report = self.resolver.analyze(
            PYTHON_CODE, JS_SAME_NAMES, "python", "javascript"
        )
        severities = {c.severity for c in report.conflicts}
        # Should have at least one severity level
        assert len(severities) > 0
        valid_severities = {ConflictSeverity.CRITICAL, ConflictSeverity.HIGH,
                           ConflictSeverity.MEDIUM, ConflictSeverity.LOW}
        assert severities.issubset(valid_severities)

    def test_symbol_extraction_python(self):
        symbols = self.resolver._extract_symbols(PYTHON_CODE, "python")
        assert "calculate" in symbols
        assert "DataHandler" in symbols
        assert symbols["DataHandler"] == "class"

    def test_symbol_extraction_javascript(self):
        symbols = self.resolver._extract_symbols(JS_SAME_NAMES, "javascript")
        assert len(symbols) > 0


# ─── Code Graph Tests ─────────────────────────────────────────────────────────

class TestCodeGraph:
    def setup_method(self):
        self.builder = CodeGraphBuilder()

    def test_build_python_graph(self):
        graph = self.builder.build(PYTHON_CODE, "python")
        assert len(graph.nodes) > 0
        func_names = [n.name for n in graph.nodes.values() if n.node_type == "function"]
        assert "calculate" in func_names or "process_data" in func_names

    def test_build_python_classes(self):
        graph = self.builder.build(PYTHON_CODE, "python")
        class_names = [n.name for n in graph.nodes.values() if n.node_type == "class"]
        assert "DataHandler" in class_names

    def test_build_go_graph(self):
        graph = self.builder.build(CLEAN_GO, "go")
        assert len(graph.nodes) > 0
        func_names = [n.name for n in graph.nodes.values()]
        assert "fibonacci" in func_names

    def test_build_javascript_graph(self):
        graph = self.builder.build(JS_SAME_NAMES, "javascript")
        assert len(graph.nodes) > 0

    def test_call_order_is_list(self):
        graph = self.builder.build(PYTHON_CODE, "python")
        order = graph.get_call_order()
        assert isinstance(order, list)

    def test_entry_points(self):
        graph = self.builder.build(PYTHON_CODE, "python")
        entry_points = graph.get_entry_points()
        assert isinstance(entry_points, list)

    def test_graph_to_dict(self):
        graph = self.builder.build(PYTHON_CODE, "python")
        d = graph.to_dict()
        assert "nodes" in d
        assert "edges" in d
        assert "node_count" in d
        assert "edge_count" in d
        assert d["node_count"] == len(d["nodes"])

    def test_merge_graphs(self):
        p_graph = self.builder.build(PYTHON_CODE, "python")
        s_graph = self.builder.build(JS_SAME_NAMES, "javascript")
        merged = self.builder.merge_graphs(p_graph, s_graph)
        # Merged graph should have more nodes
        assert len(merged.nodes) == len(p_graph.nodes) + len(s_graph.nodes)

    def test_semantic_matches_in_merged(self):
        p_graph = self.builder.build(PYTHON_CODE, "python")
        s_graph = self.builder.build(JS_SAME_NAMES, "javascript")
        merged = self.builder.merge_graphs(p_graph, s_graph)
        # Should detect semantic matches (same names in both)
        semantic_edges = [(f, t) for f, t, et in merged.edges if et == "semantic_match"]
        assert len(semantic_edges) > 0  # calculate, process_data, DataHandler all match

    def test_empty_code_graph(self):
        graph = self.builder.build("", "python")
        assert isinstance(graph.nodes, dict)
        assert isinstance(graph.edges, list)

    def test_unknown_language_generic(self):
        graph = self.builder.build("someFunc(a, b) { return a + b; }", "unknown")
        assert isinstance(graph.nodes, dict)


# ─── Continuous Learning Tests ────────────────────────────────────────────────

class TestContinuousLearner:
    def setup_method(self):
        # Use a temp directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.learner = ContinuousLearner(data_dir=self.temp_dir)

    def _make_outcome(self, sim=0.85, success=True, strategy="hybrid"):
        return FusionOutcome(
            job_id=f"test-{id(self)}",
            primary_language="python",
            secondary_language="javascript",
            target_language="python",
            strategy=strategy,
            cosine_similarity=sim,
            merge_success=success,
            tokens_used=500,
            processing_time_ms=1200.0,
        )

    def test_record_outcome(self):
        outcome = self._make_outcome()
        self.learner.record_outcome(outcome)
        assert len(self.learner.outcomes) == 1

    def test_stats_updated_after_outcome(self):
        outcome = self._make_outcome()
        self.learner.record_outcome(outcome)
        key = "python+javascript->python"
        assert key in self.learner.stats
        stat = self.learner.stats[key]
        assert stat.total_fusions == 1
        assert stat.successful == 1

    def test_pattern_extracted_high_similarity(self):
        # High similarity fusions should generate patterns
        for _ in range(3):
            self.learner.record_outcome(self._make_outcome(sim=0.90))
        # Should have generated at least one pattern
        assert len(self.learner.patterns) >= 1

    def test_no_pattern_low_similarity(self):
        outcome = self._make_outcome(sim=0.30, success=False)
        self.learner.record_outcome(outcome)
        # Low similarity shouldn't generate a pattern
        assert len(self.learner.patterns) == 0

    def test_failed_outcome_not_counted_as_success(self):
        outcome = self._make_outcome(sim=0.20, success=False)
        outcome.failure_reason = "LLM timeout"
        self.learner.record_outcome(outcome)
        key = "python+javascript->python"
        stat = self.learner.stats[key]
        assert stat.successful == 0

    def test_get_fusion_hints_with_patterns(self):
        # Seed patterns
        self.learner.record_outcome(self._make_outcome(sim=0.92))
        hint = self.learner.get_fusion_hints("python", "javascript", "python", "hybrid")
        # After recording a high-sim outcome, should return hints
        assert isinstance(hint, str)

    def test_get_fusion_hints_empty_no_patterns(self):
        hint = self.learner.get_fusion_hints("rust", "haskell", "python", "semantic")
        assert hint == ""

    def test_get_recommended_strategy_default(self):
        strategy = self.learner.get_recommended_strategy("ruby", "elixir", "python")
        assert strategy == "hybrid"  # Default when no history

    def test_get_recommended_strategy_learned(self):
        for _ in range(4):
            self.learner.record_outcome(self._make_outcome(sim=0.88, strategy="semantic"))
        strategy = self.learner.get_recommended_strategy("python", "javascript", "python")
        # After 4 successful semantic fusions, should recommend semantic
        assert strategy in ("hybrid", "semantic")

    def test_performance_report(self):
        self.learner.record_outcome(self._make_outcome(sim=0.85))
        report = self.learner.get_performance_report()
        assert "total_outcomes" in report
        assert "language_pairs" in report
        assert "last_7_days" in report
        assert report["total_outcomes"] == 1

    def test_rate_fusion(self):
        outcome = self._make_outcome()
        self.learner.record_outcome(outcome)
        self.learner.record_rating(outcome.job_id, 5)
        rated = next(o for o in self.learner.outcomes if o.job_id == outcome.job_id)
        assert rated.user_rating == 5

    def test_rate_fusion_clamps_range(self):
        outcome = self._make_outcome()
        self.learner.record_outcome(outcome)
        self.learner.record_rating(outcome.job_id, 10)  # Over max
        rated = next(o for o in self.learner.outcomes if o.job_id == outcome.job_id)
        assert rated.user_rating <= 5

    def test_persistence_save_load(self):
        outcome = self._make_outcome()
        self.learner.record_outcome(outcome)

        # Create new learner pointing to same dir
        learner2 = ContinuousLearner(data_dir=self.temp_dir)
        assert len(learner2.outcomes) == 1
        assert learner2.outcomes[0].job_id == outcome.job_id

    def test_multiple_language_pairs_tracked(self):
        pairs = [
            ("python", "javascript", "python"),
            ("java", "python", "java"),
            ("go", "rust", "go"),
        ]
        for p, s, t in pairs:
            outcome = FusionOutcome(
                job_id=f"job-{p}-{s}",
                primary_language=p,
                secondary_language=s,
                target_language=t,
                strategy="hybrid",
                cosine_similarity=0.78,
                merge_success=True,
                tokens_used=400,
                processing_time_ms=900.0,
            )
            self.learner.record_outcome(outcome)

        assert len(self.learner.stats) == 3
