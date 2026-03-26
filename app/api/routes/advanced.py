"""
app/api/routes/advanced.py - Advanced endpoints:
  - Conflict analysis
  - Code graph visualization
  - Learning feedback & reports
  - Batch fusion
  - Repository-to-repository merge
"""
import uuid
import asyncio
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from app.core.database import get_db
from app.core.schemas import Language, FusionRequest, CodeSnippet, FusionStrategy
from app.core.conflict_resolver import ConflictResolver
from app.core.code_graph import CodeGraphBuilder
from app.core.learning import get_learner
from app.agents.enhanced_pipeline import EnhancedFusionPipeline
from loguru import logger

router = APIRouter()


# ── Request/Response Models ───────────────────────────────────────────────────

class ConflictAnalysisRequest(BaseModel):
    primary_code: str = Field(..., min_length=1)
    secondary_code: str = Field(..., min_length=1)
    primary_language: Language = Language.PYTHON
    secondary_language: Language = Language.JAVASCRIPT


class GraphRequest(BaseModel):
    code: str = Field(..., min_length=1)
    language: Language = Language.PYTHON


class RatingRequest(BaseModel):
    job_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: str = Field("", max_length=500)


class BatchFusionRequest(BaseModel):
    pairs: List[FusionRequest] = Field(..., min_length=1, max_length=10)
    run_parallel: bool = Field(True, description="Run fusions in parallel")


class RepoMergeRequest(BaseModel):
    primary_files: List[CodeSnippet] = Field(..., description="Files from primary repo")
    secondary_files: List[CodeSnippet] = Field(..., description="Files from secondary repo")
    target_language: Language = Language.PYTHON
    strategy: FusionStrategy = FusionStrategy.HYBRID


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/conflicts", summary="Analyze merge conflicts before fusion")
async def analyze_conflicts(request: ConflictAnalysisRequest):
    """
    Pre-flight conflict analysis. Run this before /fuse to understand
    what conflicts exist and how they'll be resolved.
    """
    resolver = ConflictResolver()
    report = resolver.analyze(
        request.primary_code,
        request.secondary_code,
        request.primary_language.value,
        request.secondary_language.value,
    )

    return {
        "total_conflicts": report.total,
        "critical": report.critical_count,
        "auto_resolvable": report.auto_resolvable_count,
        "has_blockers": report.has_blockers,
        "summary": report.summary(),
        "conflicts": [
            {
                "type": c.type.value,
                "severity": c.severity.value,
                "symbol": c.symbol,
                "primary_context": c.primary_context,
                "secondary_context": c.secondary_context,
                "resolution": c.resolution,
                "auto_resolvable": c.auto_resolvable,
                "resolved_symbol": c.resolved_symbol,
            }
            for c in report.conflicts
        ],
        "resolution_hints": report.resolution_hints,
    }


@router.post("/graph", summary="Build code dependency graph")
async def build_code_graph(request: GraphRequest):
    """
    Build a function call graph and class hierarchy for a code snippet.
    Useful for visualizing code structure before/after fusion.
    """
    builder = CodeGraphBuilder()
    graph = builder.build(request.code, request.language.value)
    return {
        "language": request.language.value,
        **graph.to_dict(),
        "entry_points": [
            {"id": n.id, "name": n.name, "type": n.node_type}
            for n in graph.get_entry_points()
        ],
        "call_order": graph.get_call_order(),
    }


@router.post("/graph/compare", summary="Compare graphs of two code snippets")
async def compare_graphs(request: ConflictAnalysisRequest):
    """
    Build and merge the call graphs of two code snippets.
    Shows cross-language semantic matches.
    """
    builder = CodeGraphBuilder()
    p_graph = builder.build(request.primary_code, request.primary_language.value)
    s_graph = builder.build(request.secondary_code, request.secondary_language.value)
    merged = builder.merge_graphs(p_graph, s_graph)

    return {
        "primary_graph": p_graph.to_dict(),
        "secondary_graph": s_graph.to_dict(),
        "merged_graph": merged.to_dict(),
        "semantic_matches": [
            {"from": f, "to": t}
            for f, t, et in merged.edges
            if et == "semantic_match"
        ],
    }


@router.post("/fuse/enhanced", summary="Enhanced fusion with conflict resolution + learning")
async def fuse_enhanced(request: FusionRequest, db: AsyncSession = Depends(get_db)):
    """
    Production-grade fusion that includes:
    - Pre-flight conflict detection
    - Auto-resolution of naming conflicts
    - Learned prompt augmentation
    - Code graph-aware merge ordering
    - Outcome recording for continuous improvement
    """
    from app.core.database import FusionJob
    job_id = str(uuid.uuid4())

    job = FusionJob(
        id=job_id,
        status="processing",
        primary_language=request.primary.language.value,
        secondary_language=request.secondary.language.value,
        target_language=request.target_language.value,
        strategy=request.strategy.value,
        primary_code=request.primary.code,
        secondary_code=request.secondary.code,
    )
    db.add(job)
    await db.commit()

    try:
        pipeline = EnhancedFusionPipeline()
        result = await pipeline.run(request, job_id)

        job.status = result.status.value
        job.fused_code = result.fused_code
        job.explanation = result.explanation
        job.test_cases = result.test_cases
        job.agent_traces = [t.model_dump() for t in result.agent_traces]
        job.warnings = result.warnings
        if result.metrics:
            job.cosine_similarity = result.metrics.cosine_similarity
            job.processing_time_ms = result.metrics.processing_time_ms
            job.tokens_used = result.metrics.tokens_used
        await db.commit()

        return result

    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        await db.commit()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fuse/batch", summary="Batch fusion of multiple code pairs")
async def fuse_batch(request: BatchFusionRequest, db: AsyncSession = Depends(get_db)):
    """
    Fuse multiple code pairs in a single request.
    Runs in parallel (up to 10 pairs).
    """
    pipeline = EnhancedFusionPipeline()

    async def fuse_one(fusion_req: FusionRequest, idx: int):
        job_id = str(uuid.uuid4())
        try:
            result = await pipeline.run(fusion_req, job_id)
            return {"index": idx, "job_id": job_id, "status": "completed", "result": result.model_dump()}
        except Exception as e:
            return {"index": idx, "job_id": job_id, "status": "failed", "error": str(e)}

    if request.run_parallel:
        tasks = [fuse_one(req, i) for i, req in enumerate(request.pairs)]
        results = await asyncio.gather(*tasks, return_exceptions=False)
    else:
        results = []
        for i, req in enumerate(request.pairs):
            results.append(await fuse_one(req, i))

    successful = sum(1 for r in results if r["status"] == "completed")
    return {
        "total": len(results),
        "successful": successful,
        "failed": len(results) - successful,
        "results": results,
    }


@router.post("/fuse/repo", summary="Repository-level code merge")
async def fuse_repository(request: RepoMergeRequest):
    """
    Merge entire repositories by processing files intelligently.
    Groups files by type, resolves cross-file dependencies.
    """
    pipeline = EnhancedFusionPipeline()
    job_id = str(uuid.uuid4())
    results = []

    # Match primary and secondary files by name similarity
    pairs = _match_files(request.primary_files, request.secondary_files)
    logger.info(f"Repo merge: matched {len(pairs)} file pairs")

    for primary_file, secondary_file in pairs[:5]:  # Cap at 5 for demo
        fusion_req = FusionRequest(
            primary=primary_file,
            secondary=secondary_file,
            target_language=request.target_language,
            strategy=request.strategy,
            explain=False,
        )
        try:
            result = await pipeline.run(fusion_req, f"{job_id}-{len(results)}")
            results.append({
                "primary_desc": primary_file.description,
                "secondary_desc": secondary_file.description,
                "status": "completed",
                "fused_code": result.fused_code,
                "similarity": result.metrics.cosine_similarity if result.metrics else 0,
            })
        except Exception as e:
            results.append({
                "primary_desc": primary_file.description,
                "secondary_desc": secondary_file.description,
                "status": "failed",
                "error": str(e),
            })

    return {
        "repo_job_id": job_id,
        "files_processed": len(results),
        "results": results,
    }


@router.post("/feedback/rating", summary="Rate a fusion result (1-5)")
async def rate_fusion(request: RatingRequest):
    """Provide user feedback to improve the continuous learning system."""
    learner = get_learner()
    learner.record_rating(request.job_id, request.rating)
    return {
        "job_id": request.job_id,
        "rating": request.rating,
        "message": "Thank you! Your feedback improves future fusions.",
    }


@router.get("/learning/report", summary="Continuous learning performance report")
async def learning_report():
    """View the system's self-improvement statistics."""
    learner = get_learner()
    return learner.get_performance_report()


@router.get("/learning/patterns", summary="View learned fusion patterns")
async def learning_patterns():
    """View patterns the system has learned from past fusions."""
    learner = get_learner()
    return {
        "total_patterns": len(learner.patterns),
        "patterns": [
            {
                "id": p.pattern_id,
                "pair": p.language_pair,
                "target": p.target,
                "strategy": p.strategy,
                "confidence": round(p.confidence, 3),
                "uses": p.use_count,
                "hint_preview": p.prompt_hint[:150] + "...",
            }
            for p in sorted(learner.patterns, key=lambda x: x.confidence, reverse=True)
        ],
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _match_files(
    primary: List[CodeSnippet],
    secondary: List[CodeSnippet],
) -> List[tuple]:
    """Match files from two repos by description/name similarity."""
    from difflib import SequenceMatcher

    pairs = []
    used_secondary = set()

    for pf in primary:
        best_match = None
        best_score = 0.0
        p_desc = (pf.description or "").lower()

        for i, sf in enumerate(secondary):
            if i in used_secondary:
                continue
            s_desc = (sf.description or "").lower()
            score = SequenceMatcher(None, p_desc, s_desc).ratio()
            if score > best_score:
                best_score = score
                best_match = (i, sf)

        if best_match and best_score > 0.3:
            pairs.append((pf, best_match[1]))
            used_secondary.add(best_match[0])

    return pairs
