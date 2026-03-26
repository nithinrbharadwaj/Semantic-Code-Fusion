"""
app/utils/code_utils.py - Code manipulation utilities
"""
import re
from typing import List, Dict, Tuple, Optional


def detect_language(code: str) -> str:
    """Heuristic language detection."""
    from app.parsers.ast_parser import get_parser
    return get_parser().detect_language(code)


def count_lines(code: str) -> int:
    return len(code.strip().split("\n"))


def extract_functions(code: str, language: str = "python") -> List[str]:
    """Extract function names from code."""
    patterns = {
        "python": r"(?:async\s+)?def\s+(\w+)\s*\(",
        "javascript": r"(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s*)?\(?)",
        "java": r"(?:public|private|protected|static|\s)+\w+\s+(\w+)\s*\(",
        "go": r"func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(",
        "rust": r"(?:pub\s+)?fn\s+(\w+)\s*\(",
    }
    pattern = patterns.get(language, patterns["python"])
    matches = re.findall(pattern, code)
    return [m if isinstance(m, str) else next(x for x in m if x) for m in matches]


def normalize_indentation(code: str, spaces: int = 4) -> str:
    """Normalize code indentation."""
    lines = code.expandtabs(spaces).split("\n")
    # Find minimum indentation
    min_indent = float("inf")
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            min_indent = min(min_indent, len(line) - len(stripped))
    if min_indent == float("inf"):
        min_indent = 0
    return "\n".join(line[min_indent:] if len(line) > min_indent else line for line in lines)


def strip_comments(code: str, language: str = "python") -> str:
    """Remove comments from code."""
    patterns = {
        "python": [r"#[^\n]*", r'"""[\s\S]*?"""', r"'''[\s\S]*?'''"],
        "javascript": [r"//[^\n]*", r"/\*[\s\S]*?\*/"],
        "java": [r"//[^\n]*", r"/\*[\s\S]*?\*/"],
        "go": [r"//[^\n]*", r"/\*[\s\S]*?\*/"],
    }
    result = code
    for pattern in patterns.get(language, patterns["python"]):
        result = re.sub(pattern, "", result)
    return result


def extract_imports(code: str, language: str = "python") -> List[str]:
    """Extract import statements."""
    patterns = {
        "python": r"^(?:import|from)\s+\S+.*",
        "javascript": r"^(?:import|const\s+\w+\s*=\s*require)\s+.*",
        "java": r"^import\s+[\w.]+;",
        "go": r'import\s+(?:"[^"]+"|`[^`]+`|\([^)]+\))',
    }
    pattern = patterns.get(language, patterns["python"])
    return re.findall(pattern, code, re.MULTILINE)


def deduplicate_imports(imports: List[str]) -> List[str]:
    """Remove duplicate import lines."""
    seen = set()
    result = []
    for imp in imports:
        key = imp.strip()
        if key not in seen:
            seen.add(key)
            result.append(imp)
    return result


def estimate_token_count(text: str) -> int:
    """Rough token count (1 token ≈ 4 chars)."""
    return len(text) // 4
