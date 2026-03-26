"""
app/api/routes/search.py - Semantic code search
"""
import time
from fastapi import APIRouter, Depends, HTTPException, Request
from app.core.schemas import SearchRequest, SearchResponse, IndexCodeRequest
from loguru import logger

router = APIRouter()


@router.post("/search", response_model=SearchResponse, summary="Semantic code search")
async def search_code(request: SearchRequest, req: Request):
    """Search indexed code snippets using semantic similarity."""
    start = time.perf_counter()
    vector_store = req.app.state.vector_store

    try:
        results, search_ms = await vector_store.search(
            query=request.query,
            top_k=request.top_k,
            language=request.language.value if request.language else None,
            min_similarity=request.min_similarity,
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return SearchResponse(
        query=request.query,
        results=results,
        total=len(results),
        search_time_ms=search_ms,
    )


@router.post("/index", summary="Index code snippets for search")
async def index_snippets(request: IndexCodeRequest, req: Request):
    """Add code snippets to the vector index."""
    vector_store = req.app.state.vector_store
    snippets = [
        {
            "id": None,
            "code": s.code,
            "language": s.language.value,
            "description": s.description,
            "metadata": s.metadata,
        }
        for s in request.snippets
    ]
    ids = await vector_store.upsert(snippets, namespace=request.namespace)
    await vector_store.save()
    return {"indexed": len(ids), "ids": ids, "namespace": request.namespace}
