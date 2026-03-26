"""
app/agents/pipeline.py - Multi-Agent Fusion Pipeline

Agents:
  1. AnalyzerAgent  - Understands each code snippet's purpose & structure
  2. PlannerAgent   - Designs the fusion strategy
  3. FusionAgent    - Performs the actual code merge
  4. FixerAgent     - Resolves conflicts and fixes errors
  5. TesterAgent    - Generates test cases for fused code
  6. ReviewerAgent  - Final quality review
"""
import time
import uuid
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

from openai import AsyncOpenAI
from loguru import logger

from app.config import settings
from app.core.schemas import (
    FusionRequest, FusionResult, AgentTrace, FusionMetrics,
    JobStatus, Language, FusionStrategy
)
from app.utils.metrics import compute_cosine_similarity, compute_structural_overlap
from app.utils.code_utils import detect_language, extract_functions, count_lines


# ── Groq client (OpenAI-compatible) ──────────────────────────────────────────
client = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)


# ─── Agent Base ───────────────────────────────────────────────────────────────

@dataclass
class AgentContext:
    """Shared context passed between agents."""
    request: FusionRequest
    primary_analysis: dict = field(default_factory=dict)
    secondary_analysis: dict = field(default_factory=dict)
    fusion_plan: str = ""
    fused_code: str = ""
    fixed_code: str = ""
    test_cases: str = ""
    review_notes: str = ""
    traces: List[AgentTrace] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    tokens_used: int = 0

    def add_trace(self, agent: str, action: str, result: str, duration_ms: float):
        self.traces.append(AgentTrace(
            agent=agent, action=action, result=result, duration_ms=duration_ms
        ))


async def _call_llm(
    system: str,
    user: str,
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> Tuple[str, int]:
    """Call Groq (OpenAI-compatible) and return (text, tokens_used)."""
    response = await client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = response.choices[0].message.content or ""
    tokens = response.usage.total_tokens if response.usage else 0
    return text, tokens


# ─── Agent 1: Analyzer ────────────────────────────────────────────────────────

async def analyzer_agent(ctx: AgentContext) -> AgentContext:
    """Deep analysis of both code snippets."""
    start = time.perf_counter()
    logger.info("🔍 AnalyzerAgent starting...")

    req = ctx.request
    p_lang = req.primary.language.value
    s_lang = req.secondary.language.value

    system = """You are an expert code analyst. Analyze code snippets and return a structured JSON analysis.
Always return valid JSON with these fields:
{
  "purpose": "what this code does",
  "functions": ["list of function names"],
  "dependencies": ["external libs/imports used"],
  "patterns": ["design patterns observed"],
  "complexity": "low|medium|high",
  "key_logic": "summary of core algorithm/logic",
  "potential_issues": ["any concerns"],
  "reusable_components": ["components that can be merged"]
}"""

    # Analyze primary
    p_text, p_tokens = await _call_llm(
        system,
        f"Analyze this {p_lang} code:\n\n```{p_lang}\n{req.primary.code}\n```",
        temperature=0.1,
    )
    ctx.primary_analysis = _safe_json_parse(p_text)
    ctx.tokens_used += p_tokens

    # Analyze secondary
    s_text, s_tokens = await _call_llm(
        system,
        f"Analyze this {s_lang} code:\n\n```{s_lang}\n{req.secondary.code}\n```",
        temperature=0.1,
    )
    ctx.secondary_analysis = _safe_json_parse(s_text)
    ctx.tokens_used += s_tokens

    duration = (time.perf_counter() - start) * 1000
    summary = f"Primary: {ctx.primary_analysis.get('purpose', 'N/A')} | Secondary: {ctx.secondary_analysis.get('purpose', 'N/A')}"
    ctx.add_trace("AnalyzerAgent", "analyze_both_snippets", summary, duration)
    logger.info(f"✅ AnalyzerAgent done in {duration:.0f}ms")
    return ctx


# ─── Agent 2: Planner ─────────────────────────────────────────────────────────

async def planner_agent(ctx: AgentContext) -> AgentContext:
    """Design the fusion strategy."""
    start = time.perf_counter()
    logger.info("📋 PlannerAgent starting...")

    req = ctx.request

    system = """You are a senior software architect specializing in cross-language code fusion.
Given two analyzed code snippets, design a detailed fusion plan. Be specific about:
1. Which components to keep from each source
2. How to resolve naming conflicts
3. Which design patterns to apply
4. The order of operations in the merged code
5. How to handle language-specific idioms in the target language
Return a clear, numbered fusion plan as plain text."""

    user = f"""Design a fusion plan to merge these two code snippets into {req.target_language.value}.

Strategy: {req.strategy.value}

PRIMARY CODE ({req.primary.language.value}):
{req.primary.code}

SECONDARY CODE ({req.secondary.language.value}):
{req.secondary.code}

Primary Analysis: {ctx.primary_analysis}
Secondary Analysis: {ctx.secondary_analysis}

Target Language: {req.target_language.value}
"""

    plan, tokens = await _call_llm(system, user, temperature=0.3)
    ctx.fusion_plan = plan
    ctx.tokens_used += tokens

    duration = (time.perf_counter() - start) * 1000
    ctx.add_trace("PlannerAgent", "design_fusion_plan", plan[:200] + "...", duration)
    logger.info(f"✅ PlannerAgent done in {duration:.0f}ms")
    return ctx


# ─── Agent 3: Fusion ──────────────────────────────────────────────────────────

async def fusion_agent(ctx: AgentContext) -> AgentContext:
    """Perform the actual code merge."""
    start = time.perf_counter()
    logger.info("⚡ FusionAgent starting...")

    req = ctx.request
    target = req.target_language.value

    system = f"""You are an expert {target} developer performing semantic code fusion.
You merge code from different languages into clean, idiomatic {target} code.

Rules:
- Preserve ALL functionality from both sources
- Use {target} best practices and idioms
- Add clear comments explaining fused sections
- Resolve naming conflicts with clear prefixes (primary_, secondary_) or unified names
- Return ONLY the merged {target} code, no explanations outside the code
- Include all necessary imports/dependencies at the top"""

    user = f"""Execute this fusion plan to produce merged {target} code:

FUSION PLAN:
{ctx.fusion_plan}

PRIMARY CODE ({req.primary.language.value}):
```{req.primary.language.value}
{req.primary.code}
```

SECONDARY CODE ({req.secondary.language.value}):
```{req.secondary.language.value}
{req.secondary.code}
```

Produce clean, complete, runnable {target} code that combines both:"""

    fused, tokens = await _call_llm(system, user, temperature=0.15, max_tokens=6000)
    ctx.fused_code = _extract_code_block(fused, target)
    ctx.tokens_used += tokens

    duration = (time.perf_counter() - start) * 1000
    ctx.add_trace("FusionAgent", "merge_code", f"Generated {len(ctx.fused_code)} chars of {target}", duration)
    logger.info(f"✅ FusionAgent done in {duration:.0f}ms")
    return ctx


# ─── Agent 4: Fixer ───────────────────────────────────────────────────────────

async def fixer_agent(ctx: AgentContext) -> AgentContext:
    """Fix conflicts, syntax errors, and improve the fused code."""
    start = time.perf_counter()
    logger.info("🔧 FixerAgent starting...")

    req = ctx.request
    target = req.target_language.value

    system = f"""You are an expert {target} code reviewer and debugger.
Review fused code for:
1. Syntax errors
2. Undefined variables or imports
3. Logic conflicts between merged sections
4. Performance issues
5. Style inconsistencies

Fix all issues and return ONLY the corrected, clean {target} code.
If code is already correct, return it as-is."""

    user = f"""Review and fix this fused {target} code:

```{target}
{ctx.fused_code}
```

Original sources had these patterns:
- Primary: {ctx.primary_analysis.get('patterns', [])}
- Secondary: {ctx.secondary_analysis.get('patterns', [])}

Return the fixed, production-ready {target} code:"""

    fixed, tokens = await _call_llm(system, user, temperature=0.1, max_tokens=6000)
    ctx.fixed_code = _extract_code_block(fixed, target)
    ctx.tokens_used += tokens

    # Detect if fixes were needed
    if ctx.fixed_code != ctx.fused_code:
        ctx.warnings.append("FixerAgent applied corrections to the fused code")

    duration = (time.perf_counter() - start) * 1000
    ctx.add_trace("FixerAgent", "fix_and_polish", f"Fixed code: {len(ctx.fixed_code)} chars", duration)
    logger.info(f"✅ FixerAgent done in {duration:.0f}ms")
    return ctx


# ─── Agent 5: Tester ─────────────────────────────────────────────────────────

async def tester_agent(ctx: AgentContext) -> AgentContext:
    """Generate test cases for the fused code."""
    if not ctx.request.run_tests:
        ctx.add_trace("TesterAgent", "skip", "Test generation disabled", 0)
        return ctx

    start = time.perf_counter()
    logger.info("🧪 TesterAgent starting...")

    target = ctx.request.target_language.value

    system = f"""You are an expert {target} test engineer.
Generate comprehensive unit tests for the given code using the standard testing framework for {target}:
- Python → pytest
- JavaScript/TypeScript → Jest
- Java → JUnit 5
- Go → testing package

Cover: happy paths, edge cases, error handling, boundary conditions."""

    user = f"""Generate unit tests for this {target} code:

```{target}
{ctx.fixed_code}
```

Generate complete, runnable test file:"""

    tests, tokens = await _call_llm(system, user, temperature=0.2, max_tokens=3000)
    ctx.test_cases = _extract_code_block(tests, target)
    ctx.tokens_used += tokens

    duration = (time.perf_counter() - start) * 1000
    ctx.add_trace("TesterAgent", "generate_tests", f"Generated {len(ctx.test_cases)} chars of tests", duration)
    logger.info(f"✅ TesterAgent done in {duration:.0f}ms")
    return ctx


# ─── Agent 6: Reviewer ────────────────────────────────────────────────────────

async def reviewer_agent(ctx: AgentContext) -> AgentContext:
    """Final explainability review."""
    if not ctx.request.explain:
        return ctx

    start = time.perf_counter()
    logger.info("📝 ReviewerAgent starting...")

    req = ctx.request

    system = """You are a senior developer explaining a code fusion.
Write a clear, technical explanation of:
1. What the fused code does (2-3 sentences)
2. What was taken from each source and why
3. Key design decisions made during fusion
4. Any trade-offs or limitations
Keep it under 300 words. Use bullet points."""

    user = f"""Explain this fusion:

PRIMARY ({req.primary.language.value}): {ctx.primary_analysis.get('purpose', 'N/A')}
SECONDARY ({req.secondary.language.value}): {ctx.secondary_analysis.get('purpose', 'N/A')}
TARGET: {req.target_language.value}

FUSED CODE:
```{req.target_language.value}
{ctx.fixed_code[:2000]}
```

Fusion plan summary: {ctx.fusion_plan[:500]}"""

    explanation, tokens = await _call_llm(system, user, temperature=0.4, max_tokens=600)
    ctx.review_notes = explanation
    ctx.tokens_used += tokens

    duration = (time.perf_counter() - start) * 1000
    ctx.add_trace("ReviewerAgent", "explain_fusion", explanation[:200] + "...", duration)
    logger.info(f"✅ ReviewerAgent done in {duration:.0f}ms")
    return ctx


# ─── Pipeline Orchestrator ────────────────────────────────────────────────────

class FusionPipeline:
    """Orchestrates the multi-agent fusion pipeline."""

    async def run(self, request: FusionRequest, job_id: str) -> FusionResult:
        """Execute the full fusion pipeline."""
        pipeline_start = time.perf_counter()
        logger.info(f"🚀 Pipeline starting for job {job_id}")

        ctx = AgentContext(request=request)

        # Run agents sequentially
        try:
            ctx = await analyzer_agent(ctx)
            ctx = await planner_agent(ctx)
            ctx = await fusion_agent(ctx)
            ctx = await fixer_agent(ctx)
            ctx = await tester_agent(ctx)
            ctx = await reviewer_agent(ctx)
        except Exception as e:
            logger.error(f"Pipeline error in job {job_id}: {e}")
            raise

        total_ms = (time.perf_counter() - pipeline_start) * 1000

        # Auto-detect language if needed
        p_lang = request.primary.language
        s_lang = request.secondary.language
        if p_lang == Language.AUTO:
            p_lang = Language(detect_language(request.primary.code))
        if s_lang == Language.AUTO:
            s_lang = Language(detect_language(request.secondary.code))

        # Compute metrics
        sim = compute_cosine_similarity(
            request.primary.code + request.secondary.code,
            ctx.fixed_code
        )
        overlap = compute_structural_overlap(request.primary.code, ctx.fixed_code)
        p_lines = count_lines(request.primary.code) + count_lines(request.secondary.code)
        f_lines = count_lines(ctx.fixed_code)

        metrics = FusionMetrics(
            cosine_similarity=sim,
            structural_overlap=overlap,
            merge_success_rate=1.0 if ctx.fixed_code else 0.0,
            lines_added=max(0, f_lines - p_lines),
            lines_removed=max(0, p_lines - f_lines),
            processing_time_ms=total_ms,
            tokens_used=ctx.tokens_used,
        )

        logger.info(f"✅ Pipeline complete for {job_id} in {total_ms:.0f}ms | sim={sim:.3f}")

        return FusionResult(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            fused_code=ctx.fixed_code,
            target_language=request.target_language,
            strategy=request.strategy,
            explanation=ctx.review_notes or None,
            agent_traces=ctx.traces,
            metrics=metrics,
            test_cases=ctx.test_cases or None,
            warnings=ctx.warnings,
        )


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _extract_code_block(text: str, language: str) -> str:
    """Extract code from markdown code fences."""
    import re
    # Try language-specific fence
    pattern = rf"```{language}\s*(.*?)```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Try generic fence
    match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Return raw text if no fence found
    return text.strip()


def _safe_json_parse(text: str) -> dict:
    """Safely parse JSON from LLM output."""
    import json
    import re
    try:
        # Try direct parse
        return json.loads(text)
    except Exception:
        # Try extracting JSON block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return {"purpose": text[:200], "raw": text}