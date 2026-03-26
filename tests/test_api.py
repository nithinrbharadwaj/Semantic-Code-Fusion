"""
tests/test_api.py - FastAPI endpoint integration tests
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.core.schemas import JobStatus


PYTHON_SNIPPET = "def add(a, b):\n    return a + b\n"
JS_SNIPPET = "const add = (a, b) => a + b;\nmodule.exports = { add };\n"


@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    store.initialize = AsyncMock()
    store.search = AsyncMock(return_value=([], 5.0))
    store.upsert = AsyncMock(return_value=["id-1", "id-2"])
    store.save = AsyncMock()
    store.stats = MagicMock(return_value={"total_vectors": 10})
    return store


@pytest.fixture
async def client(mock_vector_store):
    app.state.vector_store = mock_vector_store
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.mark.asyncio
class TestHealthEndpoint:
    async def test_health_check(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


@pytest.mark.asyncio
class TestAnalyzeEndpoint:
    async def test_analyze_python_code(self, client):
        response = await client.post("/api/v1/analyze", json={
            "code": PYTHON_SNIPPET,
            "language": "python",
            "checks": ["complexity", "security", "quality"]
        })
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
        assert "suggestions" in data
        assert "ast_summary" in data
        assert data["language"] == "python"

    async def test_analyze_auto_detect(self, client):
        response = await client.post("/api/v1/analyze", json={
            "code": PYTHON_SNIPPET,
            "language": "auto"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["language"] in ["python", "javascript", "java", "go", "rust", "typescript"]

    async def test_analyze_empty_code(self, client):
        response = await client.post("/api/v1/analyze", json={
            "code": "",
            "language": "python"
        })
        assert response.status_code == 422  # Validation error

    async def test_analyze_security_issues(self, client):
        insecure = "import pickle\npassword = 'secret123'\ndef run(x): import os; os.system(x)\n"
        response = await client.post("/api/v1/analyze", json={
            "code": insecure,
            "language": "python"
        })
        assert response.status_code == 200
        data = response.json()
        # Should detect security issues
        assert len(data["metrics"]["security_issues"]) > 0


@pytest.mark.asyncio
class TestSearchEndpoint:
    async def test_search_returns_results(self, client, mock_vector_store):
        from app.core.schemas import SearchResult, Language
        mock_vector_store.search = AsyncMock(return_value=(
            [SearchResult(
                id="test-id",
                code="def add(a, b): return a + b",
                language=Language.PYTHON,
                similarity=0.92,
                description="Add two numbers",
            )],
            12.5
        ))

        response = await client.post("/api/v1/search", json={
            "query": "function that adds two numbers",
            "top_k": 5,
            "min_similarity": 0.5
        })
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["results"][0]["similarity"] == 0.92

    async def test_search_empty_results(self, client):
        response = await client.post("/api/v1/search", json={
            "query": "some obscure query",
            "top_k": 3,
            "min_similarity": 0.99  # Very high threshold
        })
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0

    async def test_search_too_short_query(self, client):
        response = await client.post("/api/v1/search", json={
            "query": "ab",  # Too short
            "top_k": 5
        })
        assert response.status_code == 422


@pytest.mark.asyncio
class TestIndexEndpoint:
    async def test_index_snippets(self, client):
        response = await client.post("/api/v1/index", json={
            "snippets": [
                {"code": PYTHON_SNIPPET, "language": "python", "description": "Add function"},
                {"code": JS_SNIPPET, "language": "javascript", "description": "JS add function"},
            ],
            "namespace": "test"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["indexed"] == 2
        assert len(data["ids"]) == 2


@pytest.mark.asyncio
class TestFusionEndpoint:
    @patch("app.api.routes.fusion.FusionPipeline")
    async def test_fuse_sync(self, mock_pipeline_cls, client):
        from app.core.schemas import FusionResult, FusionMetrics, JobStatus, Language, FusionStrategy
        from datetime import datetime

        mock_result = FusionResult(
            job_id="test-job-123",
            status=JobStatus.COMPLETED,
            fused_code="def add(a, b):\n    return a + b\n",
            target_language=Language.PYTHON,
            strategy=FusionStrategy.HYBRID,
            explanation="Combined Python and JS add functions",
            metrics=FusionMetrics(
                cosine_similarity=0.87,
                structural_overlap=0.6,
                merge_success_rate=1.0,
                lines_added=2,
                lines_removed=1,
                processing_time_ms=1250.0,
                tokens_used=800,
            ),
        )

        mock_instance = MagicMock()
        mock_instance.run = AsyncMock(return_value=mock_result)
        mock_pipeline_cls.return_value = mock_instance

        response = await client.post("/api/v1/fuse", json={
            "primary": {"code": PYTHON_SNIPPET, "language": "python"},
            "secondary": {"code": JS_SNIPPET, "language": "javascript"},
            "target_language": "python",
            "strategy": "hybrid",
            "explain": True,
        })

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["fused_code"] is not None
        assert data["metrics"]["cosine_similarity"] == 0.87

    async def test_fuse_validation_error(self, client):
        response = await client.post("/api/v1/fuse", json={
            "primary": {"code": "", "language": "python"},  # Empty code
            "secondary": {"code": JS_SNIPPET, "language": "javascript"},
            "target_language": "python",
        })
        assert response.status_code == 422


@pytest.mark.asyncio
class TestMetricsEndpoint:
    async def test_get_metrics(self, client):
        response = await client.get("/api/v1/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_fusions" in data
        assert "success_rate" in data
        assert "supported_languages" in data
        assert "uptime_seconds" in data
        assert len(data["supported_languages"]) > 0
