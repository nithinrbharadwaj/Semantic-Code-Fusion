"""
app/utils/security_scanner.py - Static security analysis for code
"""
import re
from typing import List, Tuple
from app.core.schemas import SecurityIssue, SecurityLevel, QualityMetrics


# ─── Vulnerability Patterns ───────────────────────────────────────────────────

SECURITY_RULES = [
    # (pattern, language_hint, severity, issue, recommendation)
    (r"eval\s*\(", "python/js", SecurityLevel.HIGH,
     "Use of eval() — code injection risk",
     "Avoid eval(); use ast.literal_eval() for Python or JSON.parse() for JS"),

    (r"exec\s*\(", "python", SecurityLevel.HIGH,
     "Use of exec() — code injection risk",
     "Avoid exec(); refactor to use functions or importlib"),

    (r"subprocess\.call\(.*shell\s*=\s*True", "python", SecurityLevel.HIGH,
     "subprocess with shell=True — command injection risk",
     "Use shell=False and pass arguments as list"),

    (r"os\.system\s*\(", "python", SecurityLevel.MEDIUM,
     "os.system() — prefer subprocess",
     "Use subprocess.run() with explicit argument list"),

    (r"pickle\.loads?\s*\(", "python", SecurityLevel.HIGH,
     "Unsafe deserialization with pickle",
     "Use JSON or other safe serialization formats"),

    (r"password\s*=\s*['\"][^'\"]{1,50}['\"]", "all", SecurityLevel.HIGH,
     "Hardcoded password detected",
     "Use environment variables or secret management systems"),

    (r"(?:api_key|secret_key|token)\s*=\s*['\"][a-zA-Z0-9_\-]{10,}['\"]",
     "all", SecurityLevel.HIGH,
     "Hardcoded API key or secret",
     "Load secrets from environment variables or a vault"),

    (r"SELECT\s+\*\s+FROM\s+\w+\s+WHERE.*%s|SELECT.*\+.*FROM", "sql", SecurityLevel.HIGH,
     "Potential SQL injection via string formatting",
     "Use parameterized queries or ORM"),

    (r"innerHTML\s*=\s*", "javascript", SecurityLevel.MEDIUM,
     "innerHTML assignment — XSS risk",
     "Use textContent or DOMPurify to sanitize HTML"),

    (r"document\.write\s*\(", "javascript", SecurityLevel.MEDIUM,
     "document.write() — XSS risk",
     "Use DOM manipulation methods instead"),

    (r"Math\.random\(\)", "javascript", SecurityLevel.LOW,
     "Math.random() is not cryptographically secure",
     "Use crypto.getRandomValues() for security-sensitive operations"),

    (r"random\.random\(\)|random\.randint\(", "python", SecurityLevel.LOW,
     "random module is not cryptographically secure",
     "Use secrets module for security-sensitive random values"),

    (r"verify\s*=\s*False", "python", SecurityLevel.MEDIUM,
     "SSL certificate verification disabled",
     "Always verify SSL certificates in production"),

    (r"DEBUG\s*=\s*True", "python", SecurityLevel.LOW,
     "Debug mode enabled",
     "Disable DEBUG in production environments"),

    (r"(?:md5|sha1)\s*\(", "python", SecurityLevel.MEDIUM,
     "Weak cryptographic hash (MD5/SHA1)",
     "Use SHA-256 or stronger via hashlib.sha256()"),
]


def scan_code(code: str, language: str = "python") -> List[SecurityIssue]:
    """Run static security scan on code."""
    issues = []
    lines = code.split("\n")

    for pattern, lang_hint, severity, issue_desc, recommendation in SECURITY_RULES:
        # Skip language-specific rules for non-matching languages
        if lang_hint not in ("all",) and language not in lang_hint:
            continue

        for match in re.finditer(pattern, code, re.IGNORECASE):
            line_num = code[:match.start()].count("\n") + 1
            issues.append(SecurityIssue(
                severity=severity,
                line=line_num,
                issue=issue_desc,
                recommendation=recommendation,
            ))

    return issues


def compute_quality_metrics(code: str, language: str = "python") -> QualityMetrics:
    """Compute overall quality metrics for code."""
    lines = code.split("\n")
    non_empty_lines = [l for l in lines if l.strip()]
    comment_lines = [l for l in non_empty_lines if _is_comment(l, language)]

    loc = len(non_empty_lines)
    comment_ratio = len(comment_lines) / max(loc, 1)

    # Cyclomatic complexity proxy
    complexity_keywords = ["if", "elif", "else", "for", "while", "try",
                           "except", "with", "case", "catch", "switch"]
    cc = 1 + sum(
        len(re.findall(rf"\b{kw}\b", code)) for kw in complexity_keywords
    )
    cc_normalized = min(cc / max(loc / 10, 1), 10.0)

    # Code duplication (simple: repeated line blocks)
    dup_score = _estimate_duplication(code)

    # Security issues
    security_issues = scan_code(code, language)
    security_penalty = sum(
        {"high": 20, "medium": 10, "low": 5}[i.severity.value]
        for i in security_issues
    )

    # Maintainability index (simplified Halstead-inspired)
    mi = max(0, min(100,
        100
        - (cc_normalized * 5)
        - (dup_score * 20)
        - security_penalty
        + (comment_ratio * 10)
    ))

    return QualityMetrics(
        cyclomatic_complexity=round(cc_normalized, 2),
        lines_of_code=loc,
        comment_ratio=round(comment_ratio, 3),
        duplication_score=round(dup_score, 3),
        maintainability_index=round(mi, 1),
        security_issues=security_issues,
        overall_score=round(mi, 1),
    )


def _is_comment(line: str, language: str) -> bool:
    stripped = line.strip()
    if language in ("python",):
        return stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''")
    return stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*")


def _estimate_duplication(code: str, min_line_len: int = 30, block_size: int = 4) -> float:
    """Simple line-level duplication detection."""
    lines = [l.strip() for l in code.split("\n") if len(l.strip()) > min_line_len]
    if len(lines) < block_size * 2:
        return 0.0

    seen = set()
    duplicates = 0
    for i in range(len(lines) - block_size + 1):
        block = tuple(lines[i:i + block_size])
        if block in seen:
            duplicates += 1
        else:
            seen.add(block)

    return min(duplicates / max(len(lines), 1), 1.0)
