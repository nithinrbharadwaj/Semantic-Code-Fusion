"""
app/core/schemas.py - Pydantic request/response models
"""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ─── Enums ────────────────────────────────────────────────────────────────────

class Language(str, Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    CPP = "cpp"
    CSHARP = "csharp"
    AUTO = "auto"  # Auto-detect


class FusionStrategy(str, Enum):
    SEMANTIC = "semantic"       # Deep semantic merge
    STRUCTURAL = "structural"   # AST-level merge
    HYBRID = "hybrid"           # Best of both (default)
    MIGRATION = "migration"     # Full language migration


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SecurityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ─── Request Models ───────────────────────────────────────────────────────────

class CodeSnippet(BaseModel):
    code: str = Field(..., description="Source code content", min_length=1)
    language: Language = Field(Language.AUTO, description="Programming language")
    description: Optional[str] = Field(None, description="What this code does")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("code")
    @classmethod
    def validate_code_length(cls, v):
        if len(v) > 50000:
            raise ValueError("Code exceeds maximum length of 50,000 characters")
        return v.strip()


class FusionRequest(BaseModel):
    primary: CodeSnippet = Field(..., description="Primary/base code snippet")
    secondary: CodeSnippet = Field(..., description="Secondary code to merge")
    target_language: Language = Field(Language.PYTHON, description="Output language")
    strategy: FusionStrategy = Field(FusionStrategy.HYBRID, description="Merge strategy")
    preserve_comments: bool = Field(True, description="Keep comments in output")
    run_tests: bool = Field(False, description="Auto-generate and run tests")
    explain: bool = Field(True, description="Include AI explanation of fusion")
    options: Dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural language or code query", min_length=3)
    language: Optional[Language] = Field(None, description="Filter by language")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")
    min_similarity: float = Field(0.5, ge=0.0, le=1.0)


class AnalyzeRequest(BaseModel):
    code: str = Field(..., min_length=1)
    language: Language = Field(Language.AUTO)
    checks: List[str] = Field(
        default=["complexity", "duplication", "security", "quality"],
        description="Which analyses to run"
    )


class MigrationRequest(BaseModel):
    code: str = Field(..., min_length=1)
    source_language: Language
    target_language: Language
    style: str = Field("idiomatic", description="Migration style: idiomatic, literal, optimized")


class IndexCodeRequest(BaseModel):
    snippets: List[CodeSnippet]
    namespace: str = Field("default", description="Vector namespace")


# ─── Response Models ──────────────────────────────────────────────────────────

class ASTNode(BaseModel):
    type: str
    name: Optional[str] = None
    start_line: int
    end_line: int
    children: List["ASTNode"] = []


class SecurityIssue(BaseModel):
    severity: SecurityLevel
    line: int
    issue: str
    recommendation: str


class QualityMetrics(BaseModel):
    cyclomatic_complexity: float
    lines_of_code: int
    comment_ratio: float
    duplication_score: float
    maintainability_index: float
    security_issues: List[SecurityIssue] = []
    overall_score: float  # 0-100


class FusionMetrics(BaseModel):
    cosine_similarity: float = Field(..., description="Semantic similarity to sources")
    structural_overlap: float
    merge_success_rate: float
    lines_added: int
    lines_removed: int
    processing_time_ms: float
    tokens_used: int


class AgentTrace(BaseModel):
    agent: str
    action: str
    result: str
    duration_ms: float


class FusionResult(BaseModel):
    job_id: str
    status: JobStatus
    fused_code: Optional[str] = None
    target_language: Language
    strategy: FusionStrategy
    explanation: Optional[str] = None
    agent_traces: List[AgentTrace] = []
    metrics: Optional[FusionMetrics] = None
    quality: Optional[QualityMetrics] = None
    test_cases: Optional[str] = None
    warnings: List[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SearchResult(BaseModel):
    id: str
    code: str
    language: Language
    similarity: float
    description: Optional[str] = None
    metadata: Dict[str, Any] = {}


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total: int
    search_time_ms: float


class AnalyzeResponse(BaseModel):
    language: Language
    ast_summary: Dict[str, Any]
    metrics: QualityMetrics
    suggestions: List[str]
    analysis_time_ms: float


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: int = Field(0, ge=0, le=100)
    result: Optional[FusionResult] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class SystemMetrics(BaseModel):
    total_fusions: int
    successful_fusions: int
    success_rate: float
    avg_processing_time_ms: float
    avg_cosine_similarity: float
    total_indexed_snippets: int
    supported_languages: List[str]
    uptime_seconds: float
