"""
app/api/routes/search.py - Semantic code search endpoints
"""
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from loguru import logger

from app.vector.store import vector_store

router = APIRouter()


# ── Request / Response models ─────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResultItem(BaseModel):
    id: str
    code: str
    language: str
    description: str
    similarity: float


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]
    total: int


class IndexSnippet(BaseModel):
    code: str
    language: str = "python"
    description: str = ""


class IndexRequest(BaseModel):
    snippets: List[IndexSnippet]
    namespace: str = "default"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/search", response_model=SearchResponse, summary="Semantic code search")
def search_code(request: SearchRequest):
    """Search indexed code snippets using semantic similarity."""
    try:
        raw_results = vector_store.search(request.query, top_k=request.top_k)

        formatted = []
        for r in raw_results:
            formatted.append(SearchResultItem(
                id=str(r.get("id") or uuid.uuid4()),
                code=r.get("code") or "",
                language=r.get("language") or "unknown",
                description=r.get("description") or "",
                similarity=float(r.get("similarity", 0.0)),
            ))

        return SearchResponse(
            query=request.query,
            results=formatted,
            total=len(formatted),
        )

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index", summary="Index code snippets for search")
def index_snippets(request: IndexRequest):
    """Add code snippets to the vector index."""
    try:
        snippets = [
            {
                "id": str(uuid.uuid4()),
                "code": s.code,
                "language": s.language,
                "description": s.description,
            }
            for s in request.snippets
        ]

        ids = vector_store.upsert(snippets)

        return {
            "indexed":   len(ids),
            "ids":       ids,
            "namespace": request.namespace,
        }

    except Exception as e:
        logger.error(f"Index error: {e}")
        raise HTTPException(status_code=500, detail=str(e))