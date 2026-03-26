"""
app/api/routes/jobs.py - Job status and management endpoints
"""
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.core.database import get_db, FusionJob
from app.core.schemas import JobStatusResponse, JobStatus, FusionResult, FusionMetrics, AgentTrace, Language, FusionStrategy

router = APIRouter()


@router.get("/job/{job_id}", response_model=JobStatusResponse, summary="Get fusion job status")
async def get_job_status(job_id: str, db: AsyncSession = Depends(get_db)):
    """Poll this endpoint to check async fusion job progress."""
    job = await db.get(FusionJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    result = None
    if job.status == "completed" and job.fused_code:
        metrics = None
        if job.cosine_similarity is not None:
            metrics = FusionMetrics(
                cosine_similarity=job.cosine_similarity or 0.0,
                structural_overlap=job.structural_overlap or 0.0,
                merge_success_rate=1.0,
                lines_added=0,
                lines_removed=0,
                processing_time_ms=job.processing_time_ms or 0.0,
                tokens_used=job.tokens_used or 0,
            )
        traces = [AgentTrace(**t) for t in (job.agent_traces or [])]
        result = FusionResult(
            job_id=job.id,
            status=JobStatus(job.status),
            fused_code=job.fused_code,
            target_language=Language(job.target_language),
            strategy=FusionStrategy(job.strategy),
            explanation=job.explanation,
            agent_traces=traces,
            metrics=metrics,
            test_cases=job.test_cases,
            warnings=job.warnings or [],
            created_at=job.created_at,
        )

    return JobStatusResponse(
        job_id=job.id,
        status=JobStatus(job.status),
        progress=job.progress or (100 if job.status == "completed" else 0),
        result=result,
        error=job.error,
        created_at=job.created_at or datetime.utcnow(),
        updated_at=job.updated_at or datetime.utcnow(),
    )


@router.get("/jobs", summary="List recent fusion jobs")
async def list_jobs(
    limit: int = 20,
    status: str = None,
    db: AsyncSession = Depends(get_db),
):
    """List recent fusion jobs with optional status filter."""
    query = select(FusionJob).order_by(FusionJob.created_at.desc()).limit(limit)
    if status:
        query = query.where(FusionJob.status == status)
    result = await db.execute(query)
    jobs = result.scalars().all()
    return [
        {
            "job_id": j.id,
            "status": j.status,
            "progress": j.progress,
            "primary_language": j.primary_language,
            "secondary_language": j.secondary_language,
            "target_language": j.target_language,
            "strategy": j.strategy,
            "cosine_similarity": j.cosine_similarity,
            "processing_time_ms": j.processing_time_ms,
            "created_at": j.created_at,
        }
        for j in jobs
    ]


@router.delete("/job/{job_id}", summary="Delete a job record")
async def delete_job(job_id: str, db: AsyncSession = Depends(get_db)):
    job = await db.get(FusionJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    await db.delete(job)
    await db.commit()
    return {"deleted": job_id}
