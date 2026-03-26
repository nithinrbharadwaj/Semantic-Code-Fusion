"""
app/agents/enhanced_pipeline.py - Production pipeline with conflict resolution + learning

Extends the base pipeline with:
  - Pre-fusion conflict detection
  - Continuous learning hint injection
  - Code graph analysis
  - Post-fusion outcome recording
"""
import time
import uuid
from typing import Optional

from loguru import logger

from app.agents.pipeline import (
    FusionPipeline as BasePipeline,
    AgentContext,
    analyzer_agent,
    planner_agent,
    fusion_agent,
    fixer_agent,
    tester_agent,
    reviewer_agent,
)
from app.core.conflict_resolver import ConflictResolver
from app.core.code_graph import CodeGraphBuilder
from app.core.learning import get_learner, FusionOutcome
from app.core.schemas import (
    FusionRequest, FusionResult, AgentTrace, FusionMetrics,
    JobStatus, Language
)
from app.utils.metrics import compute_cosine_similarity, compute_structural_overlap
from app.utils.code_utils import count_lines, detect_language


class EnhancedFusionPipeline:
    """
    Production-grade fusion pipeline with:
    - Pre-flight conflict analysis
    - Learned prompt augmentation
    - Code graph awareness
    - Outcome-based learning feedback
    All LLM calls go through pipeline.py which uses Groq.
    """

    def __init__(self):
        self.conflict_resolver = ConflictResolver()
        self.graph_builder = CodeGraphBuilder()
        self.learner = get_learner()

    async def run(self, request: FusionRequest, job_id: str) -> FusionResult:
        pipeline_start = time.perf_counter()
        logger.info(f"🚀 EnhancedPipeline starting for job {job_id}")

        p_lang = request.primary.language.value
        s_lang = request.secondary.language.value
        t_lang = request.target_language.value

        # Auto-detect languages
        if p_lang == "auto":
            p_lang = detect_language(request.primary.code)
        if s_lang == "auto":
            s_lang = detect_language(request.secondary.code)

        # ── Step 1: Pre-flight conflict analysis ───────────────────────────
        conflict_start = time.perf_counter()
        conflict_report = self.conflict_resolver.analyze(
            request.primary.code,
            request.secondary.code,
            p_lang,
            s_lang,
        )
        conflict_ms = (time.perf_counter() - conflict_start) * 1000

        logger.info(f"Conflict analysis: {conflict_report.summary()} in {conflict_ms:.0f}ms")

        # Apply auto-resolutions to secondary code
        _, patched_secondary = self.conflict_resolver.apply_auto_resolutions(
            request.primary.code,
            request.secondary.code,
            conflict_report,
        )

        # Build patched request with resolved secondary
        from app.core.schemas import CodeSnippet
        from copy import deepcopy
        patched_request = deepcopy(request)
        patched_request.secondary = CodeSnippet(
            code=patched_secondary,
            language=request.secondary.language,
            description=request.secondary.description,
        )

        # ── Step 2: Code graph analysis ────────────────────────────────────
        p_graph = self.graph_builder.build(request.primary.code, p_lang)
        s_graph = self.graph_builder.build(patched_secondary, s_lang)
        merged_graph = self.graph_builder.merge_graphs(p_graph, s_graph)
        graph_data = merged_graph.to_dict()

        # ── Step 3: Inject learned hints ───────────────────────────────────
        hint = self.learner.get_fusion_hints(p_lang, s_lang, t_lang, request.strategy.value)
        recommended_strategy = self.learner.get_recommended_strategy(p_lang, s_lang, t_lang)
        if hint:
            logger.info(f"Injecting learned hint ({len(hint)} chars)")

        # ── Step 4: Run agent pipeline ─────────────────────────────────────
        ctx = AgentContext(request=patched_request)

        # Inject conflict resolution as pre-context
        if conflict_report.conflicts:
            conflict_summary = "\n".join(
                f"- [{c.severity.value.upper()}] {c.symbol}: {c.resolution}"
                for c in conflict_report.conflicts[:5]
            )
            ctx.fusion_plan = (
                f"Pre-detected conflicts (address these):\n{conflict_summary}\n\n"
                f"Learned fusion hints:\n{hint}\n\n"
            )

        # Inject call-order hint from graph
        call_order = merged_graph.get_call_order()
        if call_order:
            node_names = [merged_graph.nodes[nid].name for nid in call_order[:6] if nid in merged_graph.nodes]
            if node_names:
                ctx.fusion_plan += f"Suggested merge order (leaves first): {', '.join(node_names)}\n\n"

        try:
            ctx = await analyzer_agent(ctx)
            ctx = await planner_agent(ctx)
            ctx = await fusion_agent(ctx)
            ctx = await fixer_agent(ctx)
            ctx = await tester_agent(ctx)
            ctx = await reviewer_agent(ctx)
        except Exception as e:
            logger.error(f"Pipeline error in job {job_id}: {e}")
            self.learner.record_outcome(FusionOutcome(
                job_id=job_id,
                primary_language=p_lang,
                secondary_language=s_lang,
                target_language=t_lang,
                strategy=request.strategy.value,
                cosine_similarity=0.0,
                merge_success=False,
                tokens_used=ctx.tokens_used,
                processing_time_ms=(time.perf_counter() - pipeline_start) * 1000,
                failure_reason=str(e)[:200],
            ))
            raise

        total_ms = (time.perf_counter() - pipeline_start) * 1000

        # ── Step 5: Compute metrics ────────────────────────────────────────
        sim = compute_cosine_similarity(
            request.primary.code + request.secondary.code,
            ctx.fixed_code,
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

        # ── Step 6: Record outcome for continuous learning ─────────────────
        self.learner.record_outcome(FusionOutcome(
            job_id=job_id,
            primary_language=p_lang,
            secondary_language=s_lang,
            target_language=t_lang,
            strategy=request.strategy.value,
            cosine_similarity=sim,
            merge_success=bool(ctx.fixed_code),
            tokens_used=ctx.tokens_used,
            processing_time_ms=total_ms,
        ))

        # ── Step 7: Add conflict traces ────────────────────────────────────
        if conflict_report.conflicts:
            ctx.add_trace(
                "ConflictResolver",
                "pre_flight_analysis",
                conflict_report.summary(),
                conflict_ms,
            )

        ctx.warnings.extend(conflict_report.resolution_hints[:2])

        logger.info(
            f"✅ EnhancedPipeline complete for {job_id} "
            f"in {total_ms:.0f}ms | sim={sim:.3f} | "
            f"conflicts={conflict_report.total}"
        )

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