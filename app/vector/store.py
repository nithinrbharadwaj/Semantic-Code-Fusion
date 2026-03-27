"""
app/vector/store.py - FAISS vector store (sync, stable version)
"""
import json
import uuid
import numpy as np
from pathlib import Path
from loguru import logger


class VectorStore:
    def __init__(self):
        self.index        = None
        self.ids          = []
        self.codes        = []
        self.languages    = []
        self.descriptions = []
        self.model        = None
        self.index_path   = Path("./data/faiss_index")

    def _get_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Embedding model loaded.")
        return self.model

    def initialize(self):
        """Load existing index or create a fresh one."""
        import faiss
        self.index_path.mkdir(parents=True, exist_ok=True)
        index_file = self.index_path / "index.faiss"
        meta_file  = self.index_path / "metadata.json"

        if index_file.exists() and meta_file.exists():
            try:
                logger.info("Loading existing FAISS index...")
                self.index = faiss.read_index(str(index_file))
                with open(meta_file, "r") as f:
                    data = json.load(f)
                self.ids          = data.get("ids", [])
                self.codes        = data.get("codes", [])
                self.languages    = data.get("languages", [])
                self.descriptions = data.get("descriptions", [])

                # Safety: FAISS count must match metadata lists
                if self.index.ntotal != len(self.ids):
                    logger.warning(
                        f"Mismatch: FAISS has {self.index.ntotal} vectors but "
                        f"metadata has {len(self.ids)} entries. Resetting."
                    )
                    self._reset()
                else:
                    logger.info(f"Loaded {self.index.ntotal} vectors")

            except Exception as e:
                logger.error(f"Failed to load index: {e}. Starting fresh.")
                self._reset()
        else:
            logger.info("Creating new FAISS index...")
            self._reset()

    def _reset(self):
        import faiss
        self.index        = faiss.IndexFlatL2(384)
        self.ids          = []
        self.codes        = []
        self.languages    = []
        self.descriptions = []

    def save(self):
        if self.index is None:
            return
        try:
            import faiss
            faiss.write_index(self.index, str(self.index_path / "index.faiss"))
            with open(self.index_path / "metadata.json", "w") as f:
                json.dump({
                    "ids":          self.ids,
                    "codes":        self.codes,
                    "languages":    self.languages,
                    "descriptions": self.descriptions,
                }, f)
            logger.info(f"Saved {self.index.ntotal} vectors to disk")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def upsert(self, snippets: list) -> list:
        """Index code snippets. Each must have a 'code' key."""
        model = self._get_model()

        texts = [
            s.get("code", "") + " " + s.get("description", "")
            for s in snippets
        ]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")
        self.index.add(embeddings)

        new_ids = []
        for s in snippets:
            uid = str(s.get("id") or uuid.uuid4())
            self.ids.append(uid)
            self.codes.append(s.get("code", ""))
            self.languages.append(s.get("language", "unknown"))
            self.descriptions.append(s.get("description") or "")
            new_ids.append(uid)

        self.save()
        logger.info(f"Indexed {len(snippets)} snippets. Total: {self.index.ntotal}")
        return new_ids

    def search(self, query: str, top_k: int = 5) -> list:
        """Search for similar code. Returns list of dicts."""
        if self.index is None or self.index.ntotal == 0:
            return []

        model = self._get_model()
        query_vec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        query_vec = np.array(query_vec, dtype="float32")

        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            # Critical bounds check — prevents IndexError
            if idx < 0 or idx >= len(self.ids):
                continue
            results.append({
                "id":          self.ids[idx],
                "code":        self.codes[idx],
                "language":    self.languages[idx],
                "description": self.descriptions[idx],
                "similarity":  round(float(1 / (1 + dist)), 4),
            })

        logger.info(f"Search returned {len(results)} results")
        return results

    def stats(self) -> dict:
        return {
            "total_vectors":  self.index.ntotal if self.index else 0,
            "metadata_count": len(self.ids),
        }


# ── Global singleton (imported by search.py and main.py) ─────────────────────
vector_store = VectorStore()