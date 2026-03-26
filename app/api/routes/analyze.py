"""
app/api/routes/analyze.py - Code analysis endpoints
"""
import time
from fastapi import APIRouter, HTTPException
from app.core.schemas import AnalyzeRequest, AnalyzeResponse, Language
from app.parsers.ast_parser import get_parser
from app.utils.security_scanner import compute_quality_metrics
from app.utils.code_utils import detect_language
from loguru import logger

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse, summary="Analyze code quality")
async def analyze_code(request: AnalyzeRequest):
    """Static analysis: AST parsing, complexity, security scan, quality metrics."""
    start = time.perf_counter()

    language = request.language
    if language == Language.AUTO:
        detected = detect_language(request.code)
        language = Language(detected)

    parser = get_parser()

    try:
        ast_summary = parser.parse(request.code, language.value)
    except Exception as e:
        logger.warning(f"AST parse failed: {e}")
        ast_summary = None

    ast_data = {}
    if ast_summary:
        ast_data = {
            "functions": [{"name": f.name, "params": f.params, "line": f.start_line} for f in ast_summary.functions],
            "classes": [{"name": c.name, "methods": c.methods, "line": c.start_line} for c in ast_summary.classes],
            "imports": ast_summary.imports,
            "line_count": ast_summary.line_count,
            "complexity_estimate": ast_summary.complexity_estimate,
        }
    else:
        ast_data = {"note": "AST parsing unavailable", "line_count": len(request.code.split("\n"))}

    metrics = compute_quality_metrics(request.code, language.value)

    # Generate suggestions
    suggestions = []
    if metrics.cyclomatic_complexity > 7:
        suggestions.append("High cyclomatic complexity — consider breaking into smaller functions")
    if metrics.comment_ratio < 0.1:
        suggestions.append("Low comment ratio — add docstrings and inline comments")
    if metrics.duplication_score > 0.2:
        suggestions.append("Code duplication detected — extract shared logic into utilities")
    for issue in metrics.security_issues:
        suggestions.append(f"[Security/{issue.severity.value.upper()}] Line {issue.line}: {issue.recommendation}")
    if not suggestions:
        suggestions.append("Code looks clean! No major issues detected.")

    elapsed = (time.perf_counter() - start) * 1000

    return AnalyzeResponse(
        language=language,
        ast_summary=ast_data,
        metrics=metrics,
        suggestions=suggestions,
        analysis_time_ms=round(elapsed, 2),
    )
