"""
app/celery_app.py - Celery configuration and async tasks
"""
import asyncio
import uuid
from celery import Celery
from loguru import logger

from app.config import settings

# ─── Celery App ───────────────────────────────────────────────────────────────

celery_app = Celery(
    "semantic_code_fusion",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=3600,  # 1 hour
    task_soft_time_limit=settings.FUSION_TIMEOUT_SECONDS,
    task_time_limit=settings.FUSION_TIMEOUT_SECONDS + 30,
)


# ─── Helper ───────────────────────────────────────────────────────────────────

def _run_async(coro):
    """Run async coroutine in Celery worker (sync context)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ─── Tasks ────────────────────────────────────────────────────────────────────

@celery_app.task(bind=True, name="tasks.run_fusion")
def run_fusion_task(self, fusion_request_dict: dict, job_id: str):
    """Background task to run the full fusion pipeline."""
    from app.agents.pipeline import FusionPipeline
    from app.core.schemas import FusionRequest
    from app.core.database import AsyncSessionLocal
    from app.core.database import FusionJob

    logger.info(f"[Task] Starting fusion job {job_id}")

    async def _execute():
        request = FusionRequest(**fusion_request_dict)
        pipeline = FusionPipeline()

        # Update job status to processing
        async with AsyncSessionLocal() as db:
            job = await db.get(FusionJob, job_id)
            if job:
                job.status = "processing"
                job.progress = 10
                await db.commit()

        # Run pipeline
        result = await pipeline.run(request, job_id)

        # Update DB with result
        async with AsyncSessionLocal() as db:
            job = await db.get(FusionJob, job_id)
            if job:
                job.status = result.status.value
                job.progress = 100
                job.fused_code = result.fused_code
                job.explanation = result.explanation
                job.test_cases = result.test_cases
                job.agent_traces = [t.model_dump() for t in result.agent_traces]
                job.warnings = result.warnings
                if result.metrics:
                    job.cosine_similarity = result.metrics.cosine_similarity
                    job.structural_overlap = result.metrics.structural_overlap
                    job.processing_time_ms = result.metrics.processing_time_ms
                    job.tokens_used = result.metrics.tokens_used
                await db.commit()

        return result.model_dump()

    try:
        self.update_state(state="STARTED", meta={"progress": 5})
        result_dict = _run_async(_execute())
        return result_dict
    except Exception as e:
        logger.error(f"[Task] Job {job_id} failed: {e}")
        _run_async(_mark_job_failed(job_id, str(e)))
        raise


@celery_app.task(name="tasks.index_snippets")
def index_snippets_task(snippets: list, namespace: str = "default"):
    """Background task to index code snippets into vector store."""
    from app.vector.store import VectorStore

    async def _execute():
        store = VectorStore()
        await store.initialize()
        ids = await store.upsert(snippets, namespace=namespace)
        await store.save()
        return ids

    return _run_async(_execute())


async def _mark_job_failed(job_id: str, error: str):
    from app.core.database import AsyncSessionLocal, FusionJob
    async with AsyncSessionLocal() as db:
        job = await db.get(FusionJob, job_id)
        if job:
            job.status = "failed"
            job.error = error
            await db.commit()
