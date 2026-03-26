"""
app/core/conflict_resolver.py - Intelligent merge conflict detection and resolution

Detects naming conflicts, import duplication, type mismatches,
and structural incompatibilities between code snippets before fusion.
"""
import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from enum import Enum


class ConflictType(str, Enum):
    NAMING = "naming"           # Same function/class name
    IMPORT = "import"           # Conflicting imports
    TYPE = "type"               # Type system mismatch
    SIGNATURE = "signature"     # Same name, different params
    STRUCTURAL = "structural"   # Architecture incompatibility
    DEPENDENCY = "dependency"   # External library conflicts


class ConflictSeverity(str, Enum):
    CRITICAL = "critical"   # Will definitely break
    HIGH = "high"           # Likely to break
    MEDIUM = "medium"       # May cause issues
    LOW = "low"             # Style/preference


@dataclass
class Conflict:
    type: ConflictType
    severity: ConflictSeverity
    symbol: str                     # The conflicting name/token
    primary_context: str            # Where it appears in primary
    secondary_context: str          # Where it appears in secondary
    resolution: str                 # Suggested resolution
    auto_resolvable: bool = False
    resolved_symbol: Optional[str] = None  # What to rename to


@dataclass
class ConflictReport:
    conflicts: List[Conflict] = field(default_factory=list)
    total: int = 0
    critical_count: int = 0
    auto_resolvable_count: int = 0
    resolution_hints: List[str] = field(default_factory=list)

    def add(self, conflict: Conflict):
        self.conflicts.append(conflict)
        self.total += 1
        if conflict.severity == ConflictSeverity.CRITICAL:
            self.critical_count += 1
        if conflict.auto_resolvable:
            self.auto_resolvable_count += 1

    @property
    def has_blockers(self) -> bool:
        return self.critical_count > 0

    def summary(self) -> str:
        return (
            f"{self.total} conflicts detected "
            f"({self.critical_count} critical, "
            f"{self.auto_resolvable_count} auto-resolvable)"
        )


class ConflictResolver:
    """
    Analyzes two code snippets for merge conflicts and
    generates resolution strategies before the AI fusion step.
    """

    def analyze(
        self,
        primary_code: str,
        secondary_code: str,
        primary_lang: str,
        secondary_lang: str,
    ) -> ConflictReport:
        """Full conflict analysis between two code snippets."""
        report = ConflictReport()

        # 1. Naming conflicts
        p_symbols = self._extract_symbols(primary_code, primary_lang)
        s_symbols = self._extract_symbols(secondary_code, secondary_lang)
        self._detect_naming_conflicts(p_symbols, s_symbols, report)

        # 2. Import conflicts
        p_imports = self._extract_imports(primary_code, primary_lang)
        s_imports = self._extract_imports(secondary_code, secondary_lang)
        self._detect_import_conflicts(p_imports, s_imports, report)

        # 3. Signature conflicts
        p_funcs = self._extract_function_signatures(primary_code, primary_lang)
        s_funcs = self._extract_function_signatures(secondary_code, secondary_lang)
        self._detect_signature_conflicts(p_funcs, s_funcs, report)

        # 4. Cross-language type conflicts
        if primary_lang != secondary_lang:
            self._detect_type_conflicts(primary_code, secondary_code, primary_lang, secondary_lang, report)

        # Generate resolution hints
        report.resolution_hints = self._generate_hints(report, primary_lang)

        return report

    def apply_auto_resolutions(
        self,
        primary_code: str,
        secondary_code: str,
        report: ConflictReport,
    ) -> Tuple[str, str]:
        """Apply automatic symbol renaming for auto-resolvable conflicts."""
        modified_secondary = secondary_code

        for conflict in report.conflicts:
            if not conflict.auto_resolvable or not conflict.resolved_symbol:
                continue

            if conflict.type == ConflictType.NAMING:
                # Rename in secondary code using word boundaries
                pattern = rf'\b{re.escape(conflict.symbol)}\b'
                modified_secondary = re.sub(
                    pattern,
                    conflict.resolved_symbol,
                    modified_secondary
                )

        return primary_code, modified_secondary

    # ── Symbol Extraction ──────────────────────────────────────────────────

    def _extract_symbols(self, code: str, language: str) -> Dict[str, str]:
        """Extract {symbol_name: symbol_type} from code."""
        symbols = {}
        patterns = {
            "function": {
                "python": r"(?:async\s+)?def\s+(\w+)\s*\(",
                "javascript": r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?(?:function|\())",
                "java": r"(?:public|private|protected|static|\s)+\w[\w<>\[\]]*\s+(\w+)\s*\(",
                "go": r"func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(",
                "rust": r"(?:pub\s+)?fn\s+(\w+)",
                "typescript": r"(?:function\s+(\w+)|(?:const|let)\s+(\w+)\s*=\s*(?:async\s*)?\()",
            },
            "class": {
                "python": r"class\s+(\w+)",
                "javascript": r"class\s+(\w+)",
                "java": r"(?:public|private|abstract|final|\s)*class\s+(\w+)",
                "go": r"type\s+(\w+)\s+struct",
                "rust": r"(?:pub\s+)?struct\s+(\w+)",
                "typescript": r"(?:class|interface)\s+(\w+)",
            },
            "variable": {
                "python": r"^([A-Z_]{2,})\s*=",  # CONSTANTS only
                "javascript": r"^(?:const|let)\s+([A-Z_]{2,})\s*=",
                "java": r"(?:public|private|static|final|\s)+\w+\s+([A-Z_]{2,})\s*=",
                "go": r"(?:var|const)\s+([A-Z]\w*)",
                "rust": r"(?:pub\s+)?const\s+([A-Z_]+)",
                "typescript": r"^(?:const|let)\s+([A-Z_]{2,})\s*=",
            },
        }

        lang = language.lower()
        for sym_type, lang_patterns in patterns.items():
            pattern = lang_patterns.get(lang, lang_patterns.get("python", ""))
            if not pattern:
                continue
            for match in re.finditer(pattern, code, re.MULTILINE):
                name = next((g for g in match.groups() if g), None)
                if name and len(name) > 1:
                    symbols[name] = sym_type

        return symbols

    def _extract_imports(self, code: str, language: str) -> Set[str]:
        """Extract imported module/package names."""
        imports = set()
        patterns = {
            "python": [
                r"^import\s+(\w+)",
                r"^from\s+(\w+)",
            ],
            "javascript": [
                r"require\(['\"]([^'\"]+)['\"]",
                r"from\s+['\"]([^'\"./][^'\"]*)['\"]",
            ],
            "typescript": [
                r"from\s+['\"]([^'\"./][^'\"]*)['\"]",
            ],
            "java": [
                r"^import\s+([\w.]+);",
            ],
            "go": [
                r'"([\w./]+)"',
            ],
            "rust": [
                r"^use\s+([\w:]+)",
            ],
        }
        lang_patterns = patterns.get(language.lower(), [])
        for pattern in lang_patterns:
            for match in re.finditer(pattern, code, re.MULTILINE):
                imports.add(match.group(1).split(".")[0].split("/")[-1])
        return imports

    def _extract_function_signatures(self, code: str, language: str) -> Dict[str, List[str]]:
        """Extract {function_name: [param_names]}."""
        signatures = {}

        if language.lower() == "python":
            pattern = r"(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)"
        elif language.lower() in ("javascript", "typescript"):
            pattern = r"(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)"
        elif language.lower() == "java":
            pattern = r"\w[\w<>]*\s+(\w+)\s*\(([^)]*)\)"
        elif language.lower() == "go":
            pattern = r"func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(([^)]*)\)"
        else:
            return signatures

        for match in re.finditer(pattern, code):
            name = match.group(1)
            params_raw = match.group(2)
            params = [p.strip().split(":")[0].split(" ")[-1].strip()
                      for p in params_raw.split(",") if p.strip()]
            signatures[name] = params

        return signatures

    # ── Conflict Detection ─────────────────────────────────────────────────

    def _detect_naming_conflicts(
        self,
        primary: Dict[str, str],
        secondary: Dict[str, str],
        report: ConflictReport,
    ):
        """Detect symbols with same name in both snippets."""
        overlap = set(primary.keys()) & set(secondary.keys())
        for name in overlap:
            p_type = primary[name]
            s_type = secondary[name]
            severity = ConflictSeverity.HIGH if p_type == s_type else ConflictSeverity.MEDIUM

            resolved = f"{name}_v2" if p_type == s_type else f"secondary_{name}"

            report.add(Conflict(
                type=ConflictType.NAMING,
                severity=severity,
                symbol=name,
                primary_context=f"{p_type} in primary",
                secondary_context=f"{s_type} in secondary",
                resolution=f"Rename secondary '{name}' to '{resolved}' or unify into single implementation",
                auto_resolvable=True,
                resolved_symbol=resolved,
            ))

    def _detect_import_conflicts(
        self,
        primary: Set[str],
        secondary: Set[str],
        report: ConflictReport,
    ):
        """Detect conflicting imports (different packages, same alias)."""
        KNOWN_CONFLICTS = {
            ("requests", "aiohttp"): "Both are HTTP clients. Consider using httpx for async compatibility.",
            ("json", "orjson"): "Both are JSON parsers. orjson is faster; prefer it.",
            ("datetime", "arrow"): "Both handle dates. arrow is higher-level.",
            ("unittest", "pytest"): "Both are test frameworks. prefer pytest.",
            ("logging", "loguru"): "Both are loggers. loguru has better ergonomics.",
        }

        for (lib1, lib2), hint in KNOWN_CONFLICTS.items():
            if lib1 in primary and lib2 in secondary:
                report.add(Conflict(
                    type=ConflictType.IMPORT,
                    severity=ConflictSeverity.LOW,
                    symbol=f"{lib1} vs {lib2}",
                    primary_context=f"uses {lib1}",
                    secondary_context=f"uses {lib2}",
                    resolution=hint,
                    auto_resolvable=False,
                ))

    def _detect_signature_conflicts(
        self,
        primary: Dict[str, List[str]],
        secondary: Dict[str, List[str]],
        report: ConflictReport,
    ):
        """Same function name but different parameter counts."""
        shared = set(primary.keys()) & set(secondary.keys())
        for name in shared:
            p_params = primary[name]
            s_params = secondary[name]
            if len(p_params) != len(s_params):
                report.add(Conflict(
                    type=ConflictType.SIGNATURE,
                    severity=ConflictSeverity.HIGH,
                    symbol=name,
                    primary_context=f"{name}({', '.join(p_params)})",
                    secondary_context=f"{name}({', '.join(s_params)})",
                    resolution=(
                        f"Merge '{name}' using *args/**kwargs or create overloaded version "
                        f"with optional parameters covering both signatures"
                    ),
                    auto_resolvable=False,
                ))

    def _detect_type_conflicts(
        self,
        primary_code: str,
        secondary_code: str,
        primary_lang: str,
        secondary_lang: str,
        report: ConflictReport,
    ):
        """Cross-language type system incompatibilities."""
        TYPE_CONFLICTS = [
            {
                "langs": {"javascript", "python"},
                "pattern_a": r"\bundefined\b",
                "pattern_b": r"\bNone\b",
                "symbol": "null/None semantics",
                "resolution": "JavaScript 'undefined' and 'null' map to Python 'None'. Add explicit None checks.",
            },
            {
                "langs": {"javascript", "python"},
                "pattern_a": r"===|!==",
                "pattern_b": r"\bis\b|\bis not\b",
                "symbol": "equality operators",
                "resolution": "JS strict equality (===) maps to Python 'is' or ==. Ensure correct operator used.",
            },
            {
                "langs": {"java", "python"},
                "pattern_a": r"\bOptional<",
                "pattern_b": r"Optional\[",
                "symbol": "Optional type",
                "resolution": "Java Optional<T> → Python Optional[T] from typing. Both express nullable values.",
            },
            {
                "langs": {"go", "python"},
                "pattern_a": r"error\b",
                "pattern_b": r"raise\b|Exception",
                "symbol": "error handling",
                "resolution": "Go returns errors as values; Python uses exceptions. Wrap Go-style returns in try/except.",
            },
        ]

        p_lang = primary_lang.lower()
        s_lang = secondary_lang.lower()

        for tc in TYPE_CONFLICTS:
            if not ({p_lang, s_lang} & tc["langs"]):
                continue
            has_a = bool(re.search(tc["pattern_a"], primary_code + secondary_code))
            has_b = bool(re.search(tc["pattern_b"], primary_code + secondary_code))
            if has_a or has_b:
                report.add(Conflict(
                    type=ConflictType.TYPE,
                    severity=ConflictSeverity.MEDIUM,
                    symbol=tc["symbol"],
                    primary_context=f"{primary_lang} semantics",
                    secondary_context=f"{secondary_lang} semantics",
                    resolution=tc["resolution"],
                    auto_resolvable=False,
                ))

    def _generate_hints(self, report: ConflictReport, target_lang: str) -> List[str]:
        """Generate actionable fusion hints based on detected conflicts."""
        hints = []
        naming_conflicts = [c for c in report.conflicts if c.type == ConflictType.NAMING]
        sig_conflicts = [c for c in report.conflicts if c.type == ConflictType.SIGNATURE]
        type_conflicts = [c for c in report.conflicts if c.type == ConflictType.TYPE]

        if naming_conflicts:
            names = [c.symbol for c in naming_conflicts[:3]]
            hints.append(
                f"Rename conflicting symbols: {', '.join(names)}. "
                f"Use primary names as canonical, prefix secondary with 'legacy_' or merge logically."
            )

        if sig_conflicts:
            hints.append(
                f"Unify {len(sig_conflicts)} function signature(s) using **kwargs or "
                f"default parameter values to accommodate both calling conventions."
            )

        if type_conflicts:
            hints.append(
                f"Address {len(type_conflicts)} cross-language type issue(s). "
                f"Add type coercion wrappers at language boundary points."
            )

        if not report.conflicts:
            hints.append("No conflicts detected — proceed with direct structural merge.")

        return hints
