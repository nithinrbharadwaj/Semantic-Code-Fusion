"""
tests/test_fusion_pipeline.py - Unit tests for the fusion pipeline
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.schemas import (
    FusionRequest, CodeSnippet, Language, FusionStrategy, JobStatus
)
from app.agents.pipeline import (
    AgentContext, _extract_code_block, _safe_json_parse, FusionPipeline
)
from app.utils.metrics import (
    compute_cosine_similarity, compute_structural_overlap, compute_merge_success_rate
)
from app.utils.code_utils import detect_language, count_lines, extract_functions
from app.utils.security_scanner import scan_code, compute_quality_metrics
from app.parsers.ast_parser import MultiLanguageParser


# ─── Fixtures ─────────────────────────────────────────────────────────────────

PYTHON_CODE = """
import os
from typing import List

def calculate_fibonacci(n: int) -> List[int]:
    \"\"\"Calculate Fibonacci sequence up to n terms.\"\"\"
    if n <= 0:
        return []
    if n == 1:
        return [0]
    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[-1] + sequence[-2])
    return sequence

class MathUtils:
    def factorial(self, n: int) -> int:
        if n < 0:
            raise ValueError("Factorial of negative number")
        if n == 0:
            return 1
        return n * self.factorial(n - 1)
"""

JAVASCRIPT_CODE = """
const axios = require('axios');

async function fetchUserData(userId) {
    try {
        const response = await axios.get(`/api/users/${userId}`);
        return response.data;
    } catch (error) {
        console.error('Failed to fetch user:', error);
        throw error;
    }
}

class DataProcessor {
    constructor(config) {
        this.config = config;
        this.cache = new Map();
    }

    process(data) {
        if (this.cache.has(data.id)) {
            return this.cache.get(data.id);
        }
        const result = data.items.map(item => item * 2);
        this.cache.set(data.id, result);
        return result;
    }
}

module.exports = { fetchUserData, DataProcessor };
"""

JAVA_CODE = """
import java.util.ArrayList;
import java.util.List;

public class SortingAlgorithms {
    public static List<Integer> bubbleSort(List<Integer> arr) {
        List<Integer> list = new ArrayList<>(arr);
        int n = list.size();
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (list.get(j) > list.get(j + 1)) {
                    int temp = list.get(j);
                    list.set(j, list.get(j + 1));
                    list.set(j + 1, temp);
                }
            }
        }
        return list;
    }
}
"""

INSECURE_CODE = """
import pickle
import os
import subprocess

password = "hardcoded_password_123"
api_key = "sk-abcdefghijklmnop123456"

def run_command(user_input):
    os.system(user_input)
    subprocess.call(user_input, shell=True)

def load_data(data):
    return pickle.loads(data)

def hash_password(pwd):
    import hashlib
    return hashlib.md5(pwd.encode()).hexdigest()
"""


# ─── Utility Tests ────────────────────────────────────────────────────────────

class TestMetrics:
    def test_cosine_similarity_identical(self):
        sim = compute_cosine_similarity(PYTHON_CODE, PYTHON_CODE)
        assert sim == 1.0

    def test_cosine_similarity_different(self):
        sim = compute_cosine_similarity(PYTHON_CODE, JAVASCRIPT_CODE)
        assert 0.0 <= sim < 1.0

    def test_cosine_similarity_empty(self):
        sim = compute_cosine_similarity("", "code here")
        assert sim == 0.0

    def test_structural_overlap_identical(self):
        overlap = compute_structural_overlap(PYTHON_CODE, PYTHON_CODE)
        assert overlap == 1.0

    def test_structural_overlap_different(self):
        overlap = compute_structural_overlap(PYTHON_CODE, JAVASCRIPT_CODE)
        assert 0.0 <= overlap <= 1.0

    def test_merge_success_rate(self):
        # Merged code contains tokens from both sources
        merged = PYTHON_CODE + "\n" + JAVASCRIPT_CODE
        rate = compute_merge_success_rate(merged, PYTHON_CODE, JAVASCRIPT_CODE)
        assert rate > 0.5

    def test_merge_success_rate_empty_fused(self):
        rate = compute_merge_success_rate("", PYTHON_CODE, JAVASCRIPT_CODE)
        assert rate == 0.0


class TestCodeUtils:
    def test_detect_python(self):
        lang = detect_language(PYTHON_CODE)
        assert lang == "python"

    def test_detect_javascript(self):
        lang = detect_language(JAVASCRIPT_CODE)
        assert lang == "javascript"

    def test_detect_java(self):
        lang = detect_language(JAVA_CODE)
        assert lang == "java"

    def test_count_lines(self):
        lines = count_lines("line1\nline2\nline3")
        assert lines == 3

    def test_extract_functions_python(self):
        funcs = extract_functions(PYTHON_CODE, "python")
        assert "calculate_fibonacci" in funcs

    def test_extract_functions_empty(self):
        funcs = extract_functions("x = 1", "python")
        assert funcs == []


class TestSecurityScanner:
    def test_detects_hardcoded_password(self):
        issues = scan_code(INSECURE_CODE, "python")
        severities = [i.severity.value for i in issues]
        assert "high" in severities

    def test_detects_pickle(self):
        issues = scan_code(INSECURE_CODE, "python")
        issue_texts = [i.issue for i in issues]
        assert any("pickle" in t.lower() for t in issue_texts)

    def test_detects_shell_true(self):
        issues = scan_code(INSECURE_CODE, "python")
        issue_texts = [i.issue for i in issues]
        assert any("shell" in t.lower() for t in issue_texts)

    def test_clean_code_no_high_issues(self):
        clean = "def add(a, b):\n    return a + b\n"
        issues = scan_code(clean, "python")
        high_issues = [i for i in issues if i.severity.value == "high"]
        assert len(high_issues) == 0

    def test_quality_metrics(self):
        metrics = compute_quality_metrics(PYTHON_CODE, "python")
        assert 0 < metrics.lines_of_code < 100
        assert 0.0 <= metrics.overall_score <= 100.0
        assert metrics.cyclomatic_complexity >= 0

    def test_quality_metrics_insecure_code(self):
        metrics = compute_quality_metrics(INSECURE_CODE, "python")
        # Insecure code should have lower score
        clean_metrics = compute_quality_metrics("def add(a, b):\n    return a + b\n", "python")
        assert metrics.overall_score <= clean_metrics.overall_score


class TestASTParser:
    def setup_method(self):
        self.parser = MultiLanguageParser()

    def test_parse_python_functions(self):
        summary = self.parser._parse_regex(PYTHON_CODE, "python")
        func_names = [f.name for f in summary.functions]
        assert "calculate_fibonacci" in func_names

    def test_parse_python_classes(self):
        summary = self.parser._parse_regex(PYTHON_CODE, "python")
        class_names = [c.name for c in summary.classes]
        assert "MathUtils" in class_names

    def test_parse_javascript_functions(self):
        summary = self.parser._parse_regex(JAVASCRIPT_CODE, "javascript")
        assert summary.line_count > 0

    def test_parse_java_classes(self):
        summary = self.parser._parse_regex(JAVA_CODE, "java")
        class_names = [c.name for c in summary.classes]
        assert "SortingAlgorithms" in class_names

    def test_detect_python(self):
        lang = self.parser.detect_language(PYTHON_CODE)
        assert lang == "python"

    def test_detect_javascript(self):
        lang = self.parser.detect_language(JAVASCRIPT_CODE)
        assert lang == "javascript"

    def test_complexity_estimate(self):
        complex_code = "if x:\n  for i in range(10):\n    while True:\n      try:\n        pass\n      except:\n        pass\n"
        simple_code = "def add(a, b):\n    return a + b\n"
        assert self.parser._estimate_complexity(complex_code) > self.parser._estimate_complexity(simple_code)


# ─── Agent Pipeline Tests ─────────────────────────────────────────────────────

class TestAgentHelpers:
    def test_extract_code_block_with_fence(self):
        text = "Here is code:\n```python\ndef hello():\n    pass\n```\nDone."
        result = _extract_code_block(text, "python")
        assert "def hello():" in result
        assert "```" not in result

    def test_extract_code_block_generic_fence(self):
        text = "```\nsome code here\n```"
        result = _extract_code_block(text, "java")
        assert result == "some code here"

    def test_extract_code_block_no_fence(self):
        text = "def hello(): pass"
        result = _extract_code_block(text, "python")
        assert result == "def hello(): pass"

    def test_safe_json_parse_valid(self):
        import json
        data = {"purpose": "test", "functions": ["foo"]}
        result = _safe_json_parse(json.dumps(data))
        assert result["purpose"] == "test"

    def test_safe_json_parse_embedded(self):
        text = 'Some text before {"purpose": "testing"} some text after'
        result = _safe_json_parse(text)
        assert result.get("purpose") == "testing"

    def test_safe_json_parse_invalid(self):
        result = _safe_json_parse("not json at all!!!")
        assert "purpose" in result or "raw" in result


class TestAgentContext:
    def test_add_trace(self):
        request = FusionRequest(
            primary=CodeSnippet(code="def foo(): pass", language=Language.PYTHON),
            secondary=CodeSnippet(code="function foo() {}", language=Language.JAVASCRIPT),
            target_language=Language.PYTHON,
        )
        ctx = AgentContext(request=request)
        ctx.add_trace("TestAgent", "test_action", "test_result", 123.4)
        assert len(ctx.traces) == 1
        assert ctx.traces[0].agent == "TestAgent"
        assert ctx.traces[0].duration_ms == 123.4


# ─── Schema Tests ─────────────────────────────────────────────────────────────

class TestSchemas:
    def test_fusion_request_valid(self):
        req = FusionRequest(
            primary=CodeSnippet(code="def foo(): pass", language=Language.PYTHON),
            secondary=CodeSnippet(code="function foo() {}", language=Language.JAVASCRIPT),
            target_language=Language.PYTHON,
            strategy=FusionStrategy.HYBRID,
        )
        assert req.target_language == Language.PYTHON
        assert req.strategy == FusionStrategy.HYBRID

    def test_code_snippet_strips_whitespace(self):
        snippet = CodeSnippet(code="  def foo(): pass  ", language=Language.PYTHON)
        assert not snippet.code.startswith(" ")

    def test_code_snippet_too_long(self):
        import pytest
        with pytest.raises(Exception):
            CodeSnippet(code="x" * 60000, language=Language.PYTHON)

    def test_code_snippet_empty(self):
        import pytest
        with pytest.raises(Exception):
            CodeSnippet(code="", language=Language.PYTHON)


# ─── Integration Tests (mocked LLM) ──────────────────────────────────────────

@pytest.mark.asyncio
class TestFusionPipelineMocked:
    @pytest.fixture
    def fusion_request(self):
        return FusionRequest(
            primary=CodeSnippet(code=PYTHON_CODE, language=Language.PYTHON),
            secondary=CodeSnippet(code=JAVASCRIPT_CODE, language=Language.JAVASCRIPT),
            target_language=Language.PYTHON,
            strategy=FusionStrategy.HYBRID,
            explain=True,
            run_tests=True,
        )

    @patch("app.agents.pipeline._call_llm")
    async def test_full_pipeline_mocked(self, mock_llm, fusion_request):
        """Test full pipeline with mocked LLM calls."""
        import json

        analysis_response = json.dumps({
            "purpose": "utility functions",
            "functions": ["calculate_fibonacci", "factorial"],
            "dependencies": ["os", "typing"],
            "patterns": ["procedural", "OOP"],
            "complexity": "medium",
            "key_logic": "Mathematical computations",
            "potential_issues": [],
            "reusable_components": ["MathUtils", "DataProcessor"]
        })

        call_count = [0]
        async def mock_call(system, user, temperature=0.2, max_tokens=4096):
            call_count[0] += 1
            if call_count[0] <= 2:
                return analysis_response, 150
            elif call_count[0] == 3:
                return "1. Keep fibonacci from Python\n2. Port DataProcessor to Python\n3. Unify imports", 100
            elif call_count[0] == 4:
                return f"```python\n{PYTHON_CODE}\n# Merged JS logic\nfrom typing import Any\n\nclass DataProcessor:\n    def __init__(self, config):\n        self.config = config\n        self.cache = {{}}\n    def process(self, data):\n        return [item * 2 for item in data.get('items', [])]\n```", 500
            elif call_count[0] == 5:
                return f"```python\n{PYTHON_CODE}\nclass DataProcessor:\n    pass\n```", 300
            elif call_count[0] == 6:
                return "```python\ndef test_fibonacci():\n    assert calculate_fibonacci(5) == [0, 1, 1, 2, 3]\n```", 200
            else:
                return "Fused Python+JS code combining Fibonacci math and DataProcessor caching.", 100

        mock_llm.side_effect = mock_call

        pipeline = FusionPipeline()
        result = await pipeline.run(fusion_request, "test-job-001")

        assert result.status == JobStatus.COMPLETED
        assert result.fused_code is not None
        assert len(result.fused_code) > 0
        assert result.metrics is not None
        assert 0.0 <= result.metrics.cosine_similarity <= 1.0
        assert result.metrics.tokens_used > 0
        assert len(result.agent_traces) >= 6  # All 6 agents ran
        assert result.test_cases is not None
        assert result.explanation is not None

    @patch("app.agents.pipeline._call_llm")
    async def test_pipeline_error_handling(self, mock_llm, fusion_request):
        """Test pipeline handles LLM errors gracefully."""
        mock_llm.side_effect = Exception("OpenAI API error")

        pipeline = FusionPipeline()
        with pytest.raises(Exception, match="OpenAI API error"):
            await pipeline.run(fusion_request, "test-job-error")
