"""
app/utils/metrics.py - Cosine similarity, structural overlap, and code metrics
"""
import re
import math
from typing import List, Dict
from collections import Counter


def compute_cosine_similarity(text1: str, text2: str) -> float:
    """
    TF-based cosine similarity between two code strings.
    Used as a lightweight proxy for semantic similarity.
    """
    def tokenize(text: str) -> List[str]:
        # Split on non-alphanumeric, keep identifier-like tokens
        return re.findall(r"[a-zA-Z_]\w*|\d+", text.lower())

    tokens1 = Counter(tokenize(text1))
    tokens2 = Counter(tokenize(text2))

    if not tokens1 or not tokens2:
        return 0.0

    # Intersection
    common = set(tokens1) & set(tokens2)
    dot = sum(tokens1[t] * tokens2[t] for t in common)

    mag1 = math.sqrt(sum(v ** 2 for v in tokens1.values()))
    mag2 = math.sqrt(sum(v ** 2 for v in tokens2.values()))

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return round(dot / (mag1 * mag2), 4)


def compute_structural_overlap(code1: str, code2: str) -> float:
    """
    Measure how much structural overlap (shared function/class names) exists.
    """
    def extract_identifiers(code: str) -> set:
        return set(re.findall(r"(?:def|function|func|class|struct)\s+(\w+)", code))

    ids1 = extract_identifiers(code1)
    ids2 = extract_identifiers(code2)

    if not ids1 and not ids2:
        return 0.0
    if not ids1 or not ids2:
        return 0.0

    overlap = ids1 & ids2
    return round(len(overlap) / len(ids1 | ids2), 4)


def compute_merge_success_rate(fused_code: str, primary: str, secondary: str) -> float:
    """
    Estimate merge success: fused code should reference elements from both sources.
    """
    def get_key_tokens(code: str) -> set:
        tokens = set(re.findall(r"[a-zA-Z_]\w{2,}", code))
        # Remove common keywords
        stopwords = {"def", "class", "import", "return", "from", "self", "true",
                     "false", "null", "none", "int", "str", "bool", "list", "dict"}
        return tokens - stopwords

    p_tokens = get_key_tokens(primary)
    s_tokens = get_key_tokens(secondary)
    f_tokens = get_key_tokens(fused_code)

    if not p_tokens and not s_tokens:
        return 0.0

    p_coverage = len(p_tokens & f_tokens) / max(len(p_tokens), 1)
    s_coverage = len(s_tokens & f_tokens) / max(len(s_tokens), 1)
    return round((p_coverage + s_coverage) / 2, 4)
