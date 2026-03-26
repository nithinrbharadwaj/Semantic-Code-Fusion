"""
app/core/learning.py - Continuous Learning System

Tracks fusion outcomes, accumulates successful patterns,
and self-improves prompts based on historical performance.
Uses a feedback loop: outcome → pattern extraction → prompt refinement.
"""
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path

from loguru import logger


@dataclass
class FusionOutcome:
    job_id: str
    primary_language: str
    secondary_language: str
    target_language: str
    strategy: str
    cosine_similarity: float
    merge_success: bool
    tokens_used: int
    processing_time_ms: float
    user_rating: Optional[int] = None   # 1-5 if user provided feedback
    failure_reason: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class LanguagePairStats:
    primary: str
    secondary: str
    target: str
    total_fusions: int = 0
    successful: int = 0
    avg_similarity: float = 0.0
    avg_time_ms: float = 0.0
    best_strategy: str = "hybrid"
    common_issues: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successful / max(self.total_fusions, 1)


@dataclass
class LearnedPattern:
    pattern_id: str
    language_pair: str      # e.g. "python->javascript"
    target: str
    strategy: str
    prompt_hint: str        # Guidance to include in future fusions
    confidence: float       # 0-1 based on number of successful uses
    use_count: int = 0
    success_count: int = 0


class ContinuousLearner:
    """
    Self-improving system that learns from fusion outcomes.
    
    Workflow:
    1. Record outcome after each fusion
    2. Extract patterns from successful fusions
    3. Inject learned patterns into future fusion prompts
    4. Track per-language-pair performance
    """

    def __init__(self, data_dir: str = "./data/learning"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.outcomes: List[FusionOutcome] = []
        self.patterns: List[LearnedPattern] = []
        self.stats: Dict[str, LanguagePairStats] = {}
        self._load()

    # ── Recording ─────────────────────────────────────────────────────────

    def record_outcome(self, outcome: FusionOutcome):
        """Record a fusion outcome for learning."""
        self.outcomes.append(outcome)

        # Update language pair stats
        key = self._pair_key(outcome.primary_language, outcome.secondary_language, outcome.target_language)
        if key not in self.stats:
            self.stats[key] = LanguagePairStats(
                primary=outcome.primary_language,
                secondary=outcome.secondary_language,
                target=outcome.target_language,
            )

        stat = self.stats[key]
        stat.total_fusions += 1
        if outcome.merge_success and outcome.cosine_similarity > 0.5:
            stat.successful += 1

        # Running average similarity
        n = stat.total_fusions
        stat.avg_similarity = (stat.avg_similarity * (n - 1) + outcome.cosine_similarity) / n
        stat.avg_time_ms = (stat.avg_time_ms * (n - 1) + outcome.processing_time_ms) / n

        # Update best strategy
        if outcome.merge_success and outcome.cosine_similarity > 0.75:
            stat.best_strategy = outcome.strategy

        if outcome.failure_reason:
            if outcome.failure_reason not in stat.common_issues:
                stat.common_issues.append(outcome.failure_reason)
                stat.common_issues = stat.common_issues[-5:]  # Keep last 5

        # Extract patterns from highly successful fusions
        if outcome.cosine_similarity > 0.80 and outcome.merge_success:
            self._extract_pattern(outcome)

        self._save()
        logger.debug(f"Recorded outcome for {key}: sim={outcome.cosine_similarity:.3f}")

    def record_rating(self, job_id: str, rating: int):
        """Record user rating (1-5) for a specific fusion job."""
        for outcome in self.outcomes:
            if outcome.job_id == job_id:
                outcome.user_rating = max(1, min(5, rating))
                self._save()
                logger.info(f"Rating {rating}/5 recorded for job {job_id}")
                return
        logger.warning(f"Job {job_id} not found for rating")

    # ── Pattern Extraction ─────────────────────────────────────────────────

    def _extract_pattern(self, outcome: FusionOutcome):
        """Extract a reusable pattern from a successful fusion."""
        pair = f"{outcome.primary_language}->{outcome.secondary_language}"
        pattern_id = f"{pair}:{outcome.strategy}:{len(self.patterns)}"

        # Generate a contextual prompt hint based on the languages
        hint = self._generate_hint(
            outcome.primary_language,
            outcome.secondary_language,
            outcome.target_language,
            outcome.strategy,
        )

        # Check if a similar pattern already exists
        for p in self.patterns:
            if p.language_pair == pair and p.strategy == outcome.strategy:
                p.use_count += 1
                p.success_count += 1
                p.confidence = p.success_count / max(p.use_count, 1)
                return

        pattern = LearnedPattern(
            pattern_id=pattern_id,
            language_pair=pair,
            target=outcome.target_language,
            strategy=outcome.strategy,
            prompt_hint=hint,
            confidence=0.6,  # Start at 60% confidence
        )
        self.patterns.append(pattern)
        logger.info(f"New pattern extracted: {pattern_id}")

    def _generate_hint(self, p_lang: str, s_lang: str, target: str, strategy: str) -> str:
        """Generate a language-pair-specific fusion hint."""
        HINTS = {
            ("python", "javascript"): {
                "hybrid": (
                    "Python uses indentation and 'self'; JavaScript uses braces and 'this'. "
                    "Map JS Promises to Python async/await. Convert JS objects to Python dicts. "
                    "Map Array.map/filter to Python list comprehensions."
                ),
                "migration": (
                    "Convert JS async/await to Python asyncio coroutines. "
                    "Replace console.log with print() or logging. "
                    "Convert require() to import statements."
                ),
            },
            ("java", "python"): {
                "hybrid": (
                    "Java is statically typed; add Python type hints. "
                    "Convert Java generics (List<T>) to Python typing.List[T]. "
                    "Map Java null to Python None. Remove verbose getter/setter boilerplate."
                ),
            },
            ("go", "python"): {
                "hybrid": (
                    "Convert Go error-return pattern to Python exception handling. "
                    "Map Go goroutines to Python asyncio tasks. "
                    "Convert Go structs to Python dataclasses or Pydantic models."
                ),
            },
            ("python", "java"): {
                "migration": (
                    "Add explicit type declarations for all variables and parameters. "
                    "Convert Python list comprehensions to Java Stream API. "
                    "Wrap Python dicts in HashMap<K,V>. Add null checks where Python would use None."
                ),
            },
        }

        pair_hints = HINTS.get((p_lang, s_lang), {})
        hint = pair_hints.get(strategy, pair_hints.get("hybrid", ""))

        if not hint:
            hint = (
                f"When merging {p_lang} and {s_lang} into {target}: "
                f"normalize naming conventions, unify error handling patterns, "
                f"and ensure all imports are compatible with {target}."
            )
        return hint

    # ── Prompt Enhancement ─────────────────────────────────────────────────

    def get_fusion_hints(
        self,
        primary_lang: str,
        secondary_lang: str,
        target_lang: str,
        strategy: str,
    ) -> str:
        """
        Get accumulated hints to inject into fusion prompts.
        Returns empty string if no patterns found.
        """
        pair = f"{primary_lang}->{secondary_lang}"
        reverse_pair = f"{secondary_lang}->{primary_lang}"

        relevant = [
            p for p in self.patterns
            if p.language_pair in (pair, reverse_pair)
            and (p.strategy == strategy or strategy == "hybrid")
            and p.confidence > 0.5
        ]

        if not relevant:
            return ""

        # Sort by confidence descending
        relevant.sort(key=lambda p: p.confidence, reverse=True)
        best = relevant[0]

        # Update usage
        best.use_count += 1

        return f"\n[Learned Pattern — confidence {best.confidence:.0%}]\n{best.prompt_hint}\n"

    def get_recommended_strategy(
        self,
        primary_lang: str,
        secondary_lang: str,
        target_lang: str,
    ) -> str:
        """Recommend the best fusion strategy based on history."""
        key = self._pair_key(primary_lang, secondary_lang, target_lang)
        stat = self.stats.get(key)
        if stat and stat.total_fusions >= 3:
            return stat.best_strategy
        return "hybrid"  # Default

    def get_performance_report(self) -> Dict:
        """Generate a performance report across all language pairs."""
        report = {
            "total_outcomes": len(self.outcomes),
            "total_patterns": len(self.patterns),
            "language_pairs": {},
        }

        for key, stat in self.stats.items():
            report["language_pairs"][key] = {
                "total": stat.total_fusions,
                "success_rate": round(stat.success_rate, 3),
                "avg_similarity": round(stat.avg_similarity, 4),
                "avg_time_ms": round(stat.avg_time_ms, 0),
                "best_strategy": stat.best_strategy,
            }

        # Top patterns
        top_patterns = sorted(self.patterns, key=lambda p: p.confidence, reverse=True)[:5]
        report["top_patterns"] = [
            {
                "pair": p.language_pair,
                "strategy": p.strategy,
                "confidence": round(p.confidence, 2),
                "uses": p.use_count,
            }
            for p in top_patterns
        ]

        # Trend: last 7 days
        week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
        recent = [o for o in self.outcomes if o.timestamp >= week_ago]
        report["last_7_days"] = {
            "fusions": len(recent),
            "avg_similarity": (
                round(sum(o.cosine_similarity for o in recent) / len(recent), 4)
                if recent else 0.0
            ),
            "success_rate": (
                round(sum(1 for o in recent if o.merge_success) / len(recent), 3)
                if recent else 0.0
            ),
        }

        return report

    # ── Persistence ────────────────────────────────────────────────────────

    def _pair_key(self, p: str, s: str, t: str) -> str:
        return f"{p}+{s}->{t}"

    def _save(self):
        try:
            data = {
                "outcomes": [asdict(o) for o in self.outcomes[-500:]],  # Keep last 500
                "patterns": [asdict(p) for p in self.patterns],
                "stats": {k: asdict(v) for k, v in self.stats.items()},
            }
            (self.data_dir / "learning_data.json").write_text(
                json.dumps(data, indent=2)
            )
        except Exception as e:
            logger.error(f"Failed to save learning data: {e}")

    def _load(self):
        path = self.data_dir / "learning_data.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            self.outcomes = [FusionOutcome(**o) for o in data.get("outcomes", [])]
            self.patterns = [LearnedPattern(**p) for p in data.get("patterns", [])]
            self.stats = {
                k: LanguagePairStats(**v)
                for k, v in data.get("stats", {}).items()
            }
            logger.info(f"Loaded {len(self.outcomes)} outcomes, {len(self.patterns)} patterns")
        except Exception as e:
            logger.warning(f"Could not load learning data: {e}")


# Singleton
_learner: Optional[ContinuousLearner] = None


def get_learner() -> ContinuousLearner:
    global _learner
    if _learner is None:
        _learner = ContinuousLearner()
    return _learner
