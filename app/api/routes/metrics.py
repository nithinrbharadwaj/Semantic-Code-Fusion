"""
app/api/routes/metrics.py - System metrics and observability
"""
import time
from datetime import datetime
from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text
from app.core.database import get_db, FusionJob
from app.core.schemas import SystemMetrics, Language

router = APIRouter()
_start_time = time.time()


@router.get("/metrics", response_model=SystemMetrics, summary="System performance metrics")
async def get_metrics(req: Request, db: AsyncSession = Depends(get_db)):
    """Aggregate metrics: fusion count, success rate, avg similarity, latency."""

    # Total jobs
    total_result = await db.execute(select(func.count()).select_from(FusionJob))
    total = total_result.scalar() or 0

    # Successful
    success_result = await db.execute(
        select(func.count()).select_from(FusionJob).where(FusionJob.status == "completed")
    )
    successful = success_result.scalar() or 0

    # Avg processing time
    avg_time_result = await db.execute(
        select(func.avg(FusionJob.processing_time_ms)).where(FusionJob.processing_time_ms.isnot(None))
    )
    avg_time = avg_time_result.scalar() or 0.0

    # Avg cosine similarity
    avg_sim_result = await db.execute(
        select(func.avg(FusionJob.cosine_similarity)).where(FusionJob.cosine_similarity.isnot(None))
    )
    avg_sim = avg_sim_result.scalar() or 0.0

    # Vector store stats
    vector_store = req.app.state.vector_store
    vs_stats = vector_store.stats() if vector_store else {}

    return SystemMetrics(
        total_fusions=total,
        successful_fusions=successful,
        success_rate=round(successful / max(total, 1), 4),
        avg_processing_time_ms=round(float(avg_time), 2),
        avg_cosine_similarity=round(float(avg_sim), 4),
        total_indexed_snippets=vs_stats.get("total_vectors", 0),
        supported_languages=[l.value for l in Language if l != Language.AUTO],
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@router.get("/metrics/history", summary="Recent fusion performance history")
async def get_metrics_history(limit: int = 50, db: AsyncSession = Depends(get_db)):
    """Time-series data for dashboards."""
    result = await db.execute(
        select(
            FusionJob.created_at,
            FusionJob.cosine_similarity,
            FusionJob.processing_time_ms,
            FusionJob.status,
            FusionJob.target_language,
        )
        .order_by(FusionJob.created_at.desc())
        .limit(limit)
    )
    rows = result.fetchall()
    return [
        {
            "timestamp": row.created_at,
            "cosine_similarity": row.cosine_similarity,
            "processing_time_ms": row.processing_time_ms,
            "status": row.status,
            "target_language": row.target_language,
        }
        for row in rows
    ]
