"""
app/tasks.py - Celery async tasks
"""
import asyncio
from app.worker import celery_app


def _to_dict(obj):
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, list):
        return [_to_dict(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


def _update_job_in_db(job_id: str, status: str, result: dict = None, error: str = None):
    """Update fusion_jobs table using the real column names."""
    try:
        import json
        from sqlalchemy import create_engine, text
        from app.config import settings

        engine = create_engine(settings.SYNC_DATABASE_URL)
        with engine.connect() as conn:
            if result is not None:
                metrics = result.get("metrics") or {}
                agent_traces = result.get("agent_traces")
                conn.execute(
                    text("""
                        UPDATE fusion_jobs
                        SET status            = :status,
                            progress          = 100,
                            fused_code        = :fused_code,
                            explanation       = :explanation,
                            agent_traces      = :agent_traces,
                            cosine_similarity = :cosine_similarity,
                            processing_time_ms= :processing_time_ms,
                            tokens_used       = :tokens_used,
                            quality_score     = :quality_score,
                            updated_at        = NOW()
                        WHERE id = :job_id
                    """),
                    {
                        "status":             status,
                        "fused_code":         result.get("fused_code", ""),
                        "explanation":        result.get("explanation", ""),
                        "agent_traces":       json.dumps(agent_traces) if agent_traces else None,
                        "cosine_similarity":  metrics.get("cosine_similarity"),
                        "processing_time_ms": metrics.get("processing_time_ms") or metrics.get("duration_ms"),
                        "tokens_used":        metrics.get("tokens_used"),
                        "quality_score":      metrics.get("quality_score"),
                        "job_id":             job_id,
                    }
                )
            else:
                conn.execute(
                    text("""
                        UPDATE fusion_jobs
                        SET status     = :status,
                            error      = :error,
                            updated_at = NOW()
                        WHERE id = :job_id
                    """),
                    {"status": status, "error": error or "", "job_id": job_id}
                )
            conn.commit()
            print(f"[tasks] DB updated job {job_id} -> {status}")
    except Exception as e:
        print(f"[tasks] DB update failed: {e}")


@celery_app.task(bind=True, name="tasks.run_fusion")
def run_fusion(self, request_data: dict, job_id: str):
    try:
        from app.agents.pipeline import FusionPipeline
        from app.core.schemas import CodeSnippet, FusionRequest

        primary  = request_data.get("primary", {})
        secondary = request_data.get("secondary", {})

        req = FusionRequest(
            primary=CodeSnippet(
                code=primary.get("code", ""),
                language=primary.get("language", "python"),
            ),
            secondary=CodeSnippet(
                code=secondary.get("code", ""),
                language=secondary.get("language", "python"),
            ),
            target_language=request_data.get("target_language", "python"),
            strategy=request_data.get("strategy", "hybrid"),
            explain=request_data.get("explain", True),
            run_tests=request_data.get("run_tests", False),
        )

        pipeline = FusionPipeline()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(pipeline.run(req, job_id))
        loop.close()

        result_dict = {
            "status":       "completed",
            "fused_code":   result.fused_code,
            "explanation":  result.explanation,
            "metrics":      _to_dict(result.metrics),
            "agent_traces": _to_dict(result.agent_traces),
        }

        _update_job_in_db(job_id, "completed", result_dict)
        return result_dict

    except Exception as exc:
        error_msg = str(exc)
        _update_job_in_db(job_id, "failed", error=error_msg)
        return {"status": "failed", "error": error_msg}


@celery_app.task(bind=True, name="app.tasks.fuse_code_async")
def fuse_code_async(self, primary_code: str, primary_language: str,
                    secondary_code: str, secondary_language: str,
                    target_language: str = "python",
                    strategy: str = "hybrid",
                    explain: bool = True):
    return run_fusion(
        self,
        {
            "primary":   {"code": primary_code,   "language": primary_language},
            "secondary": {"code": secondary_code, "language": secondary_language},
            "target_language": target_language,
            "strategy":  strategy,
            "explain":   explain,
        },
        self.request.id or "",
    )