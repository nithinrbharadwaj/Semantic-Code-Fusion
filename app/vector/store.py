"""
app/vector/store.py - FAISS vector store for semantic code retrieval
Uses local sentence-transformers for embeddings (no API cost).
"""
import os
import json
import uuid
import time
import asyncio
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger

from app.config import settings
from app.core.schemas import SearchResult, Language


class VectorStore:
    """
    FAISS-backed vector store for code snippets.
    Embeddings use local sentence-transformers (all-MiniLM-L6-v2).
    Free, fast, no API key needed for search/index.
    """

    def __init__(self):
        self.index = None
        self.metadata: List[dict] = []
        self.dimension = 384          # all-MiniLM-L6-v2 output dim
        self.index_path = Path(settings.FAISS_INDEX_PATH)
        self._lock = asyncio.Lock()
        self._embed_model = None      # lazy-loaded

    def _get_embed_model(self):
        """Lazy-load the embedding model (downloads once, cached locally)."""
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
            self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Embedding model loaded.")
        return self._embed_model

    async def initialize(self):
        """Load or create FAISS index."""
        try:
            import faiss
        except ImportError:
            logger.warning("faiss-cpu not installed. Vector search disabled.")
            return

        self.index_path.mkdir(parents=True, exist_ok=True)
        index_file = self.index_path / "index.faiss"
        meta_file  = self.index_path / "metadata.json"

        if index_file.exists() and meta_file.exists():
            logger.info("Loading existing FAISS index...")
            self.index = faiss.read_index(str(index_file))
            with open(meta_file, "r") as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded {self.index.ntotal} vectors")
        else:
            logger.info("Creating new FAISS index (IndexFlatIP for cosine similarity)...")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []

    async def save(self):
        """Persist index to disk."""
        if self.index is None:
            return
        try:
            import faiss
            faiss.write_index(self.index, str(self.index_path / "index.faiss"))
            with open(self.index_path / "metadata.json", "w") as f:
                json.dump(self.metadata, f)
            logger.info(f"Saved {self.index.ntotal} vectors to disk")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    async def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings using local sentence-transformers model.
        Runs in a thread pool to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()

        def _encode():
            model = self._get_embed_model()
            embeddings = model.encode(
                texts,
                normalize_embeddings=True,   # L2-norm → cosine via inner product
                show_progress_bar=False,
            )
            return np.array(embeddings, dtype=np.float32)

        embeddings = await loop.run_in_executor(None, _encode)
        return embeddings

    async def upsert(
        self,
        snippets: List[dict],
        namespace: str = "default",
    ) -> List[str]:
        """Embed and index code snippets."""
        if self.index is None:
            logger.warning("Vector store not initialized")
            return []

        async with self._lock:
            texts = [s["code"] for s in snippets]
            embeddings = await self.embed(texts)

            ids = []
            for i, snippet in enumerate(snippets):
                sid      = snippet.get("id", str(uuid.uuid4()))
                faiss_id = self.index.ntotal + i
                self.index.add(embeddings[i : i + 1])
                self.metadata.append({
                    "id":          sid,
                    "faiss_id":    faiss_id,
                    "code":        snippet["code"],
                    "language":    snippet.get("language", "unknown"),
                    "description": snippet.get("description"),
                    "namespace":   namespace,
                    "metadata":    snippet.get("metadata", {}),
                })
                ids.append(sid)

            logger.info(f"Indexed {len(snippets)} snippets. Total: {self.index.ntotal}")
            return ids

    async def search(
        self,
        query: str,
        top_k: int = 5,
        language: Optional[str] = None,
        min_similarity: float = 0.5,
        namespace: str = "default",
    ) -> Tuple[List[SearchResult], float]:
        """Semantic search over indexed snippets."""
        if self.index is None or self.index.ntotal == 0:
            return [], 0.0

        start     = time.perf_counter()
        query_vec = await self.embed([query])

        k = min(top_k * 4, self.index.ntotal)
        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or float(score) < min_similarity:
                continue
            meta = self.metadata[idx]
            if language and meta.get("language") != language:
                continue
            if namespace and meta.get("namespace") != namespace:
                continue
            results.append(SearchResult(
                id=meta["id"],
                code=meta["code"],
                language=Language(meta.get("language", "python")),
                similarity=float(score),
                description=meta.get("description"),
                metadata=meta.get("metadata", {}),
            ))
            if len(results) >= top_k:
                break

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"Search returned {len(results)} results in {elapsed_ms:.1f}ms")
        return results, elapsed_ms

    async def delete(self, ids: List[str]):
        """Soft-delete snippets by ID."""
        async with self._lock:
            for meta in self.metadata:
                if meta["id"] in ids:
                    meta["deleted"] = True
        logger.info(f"Soft-deleted {len(ids)} snippets")

    def stats(self) -> dict:
        return {
            "total_vectors":    self.index.ntotal if self.index else 0,
            "dimension":        self.dimension,
            "metadata_count":   len(self.metadata),
        }