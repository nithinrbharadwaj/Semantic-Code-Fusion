"""
app/api/routes/fusion.py - Fusion endpoints
"""
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.database import get_db, FusionJob
from app.core.schemas import (
    FusionRequest, FusionResult, JobStatusResponse, JobStatus,
    MigrationRequest
)
from app.agents.pipeline import FusionPipeline
from app.config import settings

router = APIRouter()


# ─── Synchronous Fusion ───────────────────────────────────────────────────────

@router.post(
    "/fuse",
    response_model=FusionResult,
    summary="Fuse two code snippets (synchronous)",
    description="Runs the full multi-agent pipeline and returns result immediately.",
)
async def fuse_code(
    request: FusionRequest,
    db: AsyncSession = Depends(get_db),
):
    """Synchronous fusion — waits for full pipeline to complete."""
    job_id = str(uuid.uuid4())
    logger.info(f"Sync fusion job {job_id}: {request.primary.language} + {request.secondary.language} → {request.target_language}")

    # Save job record
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
        pipeline = FusionPipeline()
        result = await pipeline.run(request, job_id)

        # Update DB
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
        logger.error(f"Fusion job {job_id} failed: {e}")
        raise HTTPException(status_code=500, detail=f"Fusion failed: {str(e)}")


# ─── Async Fusion ─────────────────────────────────────────────────────────────

@router.post(
    "/fuse/async",
    response_model=dict,
    summary="Fuse code asynchronously (background job)",
    description="Enqueues fusion job via Celery. Poll /job/{job_id} for status.",
)
async def fuse_code_async(
    request: FusionRequest,
    db: AsyncSession = Depends(get_db),
):
    """Async fusion via Celery task queue."""
    job_id = str(uuid.uuid4())

    # Save job record
    job = FusionJob(
        id=job_id,
        status="pending",
        primary_language=request.primary.language.value,
        secondary_language=request.secondary.language.value,
        target_language=request.target_language.value,
        strategy=request.strategy.value,
        primary_code=request.primary.code,
        secondary_code=request.secondary.code,
    )
    db.add(job)
    await db.commit()

    # Enqueue Celery task
    try:
        from app.celery_app import run_fusion_task
        run_fusion_task.apply_async(
            args=[request.model_dump(), job_id],
            task_id=job_id,
        )
        logger.info(f"Async fusion job {job_id} enqueued")
    except Exception as e:
        logger.warning(f"Celery unavailable, running inline: {e}")
        # Fallback: run synchronously
        pipeline = FusionPipeline()
        result = await pipeline.run(request, job_id)
        job.status = result.status.value
        job.fused_code = result.fused_code
        await db.commit()
        return {"job_id": job_id, "status": "completed", "note": "ran synchronously"}

    return {
        "job_id": job_id,
        "status": "pending",
        "poll_url": f"/api/v1/job/{job_id}",
    }


# ─── Language Migration ───────────────────────────────────────────────────────

@router.post(
    "/migrate",
    response_model=FusionResult,
    summary="Migrate code from one language to another",
)
async def migrate_code(
    request: MigrationRequest,
    db: AsyncSession = Depends(get_db),
):
    """Full language migration with idiomatic rewrite."""
    from app.core.schemas import FusionRequest, CodeSnippet, FusionStrategy

    # Create a fusion request where secondary is empty placeholder
    fusion_req = FusionRequest(
        primary=CodeSnippet(
            code=request.code,
            language=request.source_language,
            description=f"Code to migrate from {request.source_language.value}",
        ),
        secondary=CodeSnippet(
            code=f"# Empty target — migrate primary to {request.target_language.value}",
            language=request.target_language,
            description="Migration target placeholder",
        ),
        target_language=request.target_language,
        strategy=FusionStrategy.MIGRATION,
        explain=True,
    )

    job_id = str(uuid.uuid4())
    pipeline = FusionPipeline()
    result = await pipeline.run(fusion_req, job_id)
    return result
