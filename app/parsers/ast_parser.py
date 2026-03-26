"""
app/parsers/ast_parser.py - Multi-language AST parsing with Tree-sitter
"""
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class ParsedFunction:
    name: str
    start_line: int
    end_line: int
    params: List[str]
    docstring: Optional[str] = None
    language: str = "unknown"


@dataclass
class ParsedClass:
    name: str
    start_line: int
    end_line: int
    methods: List[str]
    bases: List[str]


@dataclass
class ASTSummary:
    language: str
    functions: List[ParsedFunction]
    classes: List[ParsedClass]
    imports: List[str]
    global_vars: List[str]
    line_count: int
    complexity_estimate: float


class MultiLanguageParser:
    """
    Parses code using Tree-sitter when available,
    falls back to regex-based parsing.
    """

    LANGUAGE_PATTERNS = {
        "python": {
            "function": r"(?:^|\n)(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)",
            "class": r"(?:^|\n)class\s+(\w+)(?:\(([^)]*)\))?:",
            "import": r"^(?:import|from)\s+.+",
            "comment": r"#.*|\"\"\"[\s\S]*?\"\"\"|\'\'\'[\s\S]*?\'\'\'",
        },
        "javascript": {
            "function": r"(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|\w+)\s*=>|(\w+)\s*:\s*function)",
            "class": r"class\s+(\w+)(?:\s+extends\s+(\w+))?",
            "import": r"^(?:import|require)\s+.+",
            "comment": r"//.*|/\*[\s\S]*?\*/",
        },
        "java": {
            "function": r"(?:public|private|protected|static|\s)+[\w<>\[\]]+\s+(\w+)\s*\(([^)]*)\)",
            "class": r"(?:public|private|abstract|final|\s)*class\s+(\w+)(?:\s+extends\s+(\w+))?",
            "import": r"^import\s+.+;",
            "comment": r"//.*|/\*[\s\S]*?\*/",
        },
        "go": {
            "function": r"func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(([^)]*)\)",
            "class": r"type\s+(\w+)\s+struct",
            "import": r"import\s+(?:\([\s\S]*?\)|\"[^\"]+\")",
            "comment": r"//.*|/\*[\s\S]*?\*/",
        },
        "typescript": {
            "function": r"(?:function\s+(\w+)|(?:const|let)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|\w+)\s*=>|(\w+)\s*\(([^)]*)\)\s*:)",
            "class": r"class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+[\w,\s]+)?",
            "import": r"^import\s+.+",
            "comment": r"//.*|/\*[\s\S]*?\*/",
        },
        "rust": {
            "function": r"(?:pub\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\(([^)]*)\)",
            "class": r"(?:pub\s+)?struct\s+(\w+)|(?:pub\s+)?impl\s+(\w+)",
            "import": r"^use\s+.+;",
            "comment": r"//.*|/\*[\s\S]*?\*/",
        },
    }

    def parse(self, code: str, language: str) -> ASTSummary:
        """Parse code and return AST summary."""
        language = language.lower()

        # Try Tree-sitter first
        try:
            return self._parse_tree_sitter(code, language)
        except Exception as e:
            logger.debug(f"Tree-sitter failed for {language}: {e}, using regex fallback")

        # Regex fallback
        return self._parse_regex(code, language)

    def _parse_tree_sitter(self, code: str, language: str) -> ASTSummary:
        """Parse with Tree-sitter library."""
        from tree_sitter import Language as TSLanguage, Parser

        lang_map = {
            "python": "tree_sitter_python",
            "javascript": "tree_sitter_javascript",
            "java": "tree_sitter_java",
        }

        if language not in lang_map:
            raise ValueError(f"Tree-sitter not configured for {language}")

        module = __import__(lang_map[language])
        ts_lang = TSLanguage(module.language())
        parser = Parser(ts_lang)
        tree = parser.parse(bytes(code, "utf8"))
        root = tree.root_node

        functions = self._extract_ts_functions(root, code, language)
        classes = self._extract_ts_classes(root, code, language)
        imports = self._extract_imports_regex(code, language)

        return ASTSummary(
            language=language,
            functions=functions,
            classes=classes,
            imports=imports,
            global_vars=[],
            line_count=code.count("\n") + 1,
            complexity_estimate=self._estimate_complexity(code),
        )

    def _extract_ts_functions(self, node, code: str, language: str) -> List[ParsedFunction]:
        """Walk tree-sitter AST to extract functions."""
        functions = []

        def walk(n):
            if n.type in ("function_definition", "function_declaration", "method_definition"):
                name_node = n.child_by_field_name("name")
                params_node = n.child_by_field_name("parameters")
                if name_node:
                    params = []
                    if params_node:
                        for child in params_node.children:
                            if child.type in ("identifier", "typed_parameter", "typed_default_parameter"):
                                params.append(code[child.start_byte:child.end_byte])
                    functions.append(ParsedFunction(
                        name=code[name_node.start_byte:name_node.end_byte],
                        start_line=n.start_point[0] + 1,
                        end_line=n.end_point[0] + 1,
                        params=params,
                        language=language,
                    ))
            for child in n.children:
                walk(child)

        walk(node)
        return functions

    def _extract_ts_classes(self, node, code: str, language: str) -> List[ParsedClass]:
        """Walk tree-sitter AST to extract classes."""
        classes = []

        def walk(n):
            if n.type in ("class_definition", "class_declaration"):
                name_node = n.child_by_field_name("name")
                if name_node:
                    classes.append(ParsedClass(
                        name=code[name_node.start_byte:name_node.end_byte],
                        start_line=n.start_point[0] + 1,
                        end_line=n.end_point[0] + 1,
                        methods=[],
                        bases=[],
                    ))
            for child in n.children:
                walk(child)

        walk(node)
        return classes

    def _parse_regex(self, code: str, language: str) -> ASTSummary:
        """Regex-based parsing fallback."""
        patterns = self.LANGUAGE_PATTERNS.get(language, self.LANGUAGE_PATTERNS["python"])
        lines = code.split("\n")

        functions = []
        for match in re.finditer(patterns["function"], code, re.MULTILINE):
            name = next((g for g in match.groups() if g), "unknown")
            line_num = code[:match.start()].count("\n") + 1
            params_str = match.groups()[-1] if len(match.groups()) > 1 else ""
            params = [p.strip() for p in params_str.split(",") if p.strip()] if params_str else []
            functions.append(ParsedFunction(
                name=name, start_line=line_num, end_line=line_num + 5,
                params=params, language=language
            ))

        classes = []
        for match in re.finditer(patterns["class"], code, re.MULTILINE):
            name = match.group(1) or "unknown"
            line_num = code[:match.start()].count("\n") + 1
            bases = [match.group(2)] if len(match.groups()) > 1 and match.group(2) else []
            classes.append(ParsedClass(
                name=name, start_line=line_num, end_line=line_num + 20,
                methods=[], bases=bases
            ))

        imports = self._extract_imports_regex(code, language)

        return ASTSummary(
            language=language,
            functions=functions,
            classes=classes,
            imports=imports,
            global_vars=[],
            line_count=len(lines),
            complexity_estimate=self._estimate_complexity(code),
        )

    def _extract_imports_regex(self, code: str, language: str) -> List[str]:
        patterns = self.LANGUAGE_PATTERNS.get(language, {})
        pattern = patterns.get("import", r"^import\s+.+")
        return re.findall(pattern, code, re.MULTILINE)[:20]

    def _estimate_complexity(self, code: str) -> float:
        """McCabe-like complexity via keyword counting."""
        keywords = ["if", "elif", "else", "for", "while", "try", "except",
                    "with", "case", "catch", "switch", "&&", "||"]
        count = sum(len(re.findall(rf"\b{kw}\b", code)) for kw in keywords[:8])
        count += code.count("&&") + code.count("||") + code.count("??")
        return min(count / max(code.count("\n"), 1) * 10, 10.0)

    def detect_language(self, code: str) -> str:
        """Heuristic language detection."""
        scores: Dict[str, int] = {}

        indicators = {
            "python": ["def ", "import ", "print(", "self.", "elif ", "__init__", "->", ":#"],
            "javascript": ["const ", "let ", "var ", "=>", "console.log", "require(", ".then("],
            "typescript": ["interface ", ": string", ": number", "readonly ", "async (", "<T>"],
            "java": ["public class", "System.out", "void main", "new ", "@Override", "extends "],
            "go": ["func ", "package ", ":=", "fmt.Print", "go func", "chan ", "defer "],
            "rust": ["fn ", "let mut", "impl ", "use std::", "println!", "-> Result", "Option<"],
        }

        for lang, signs in indicators.items():
            scores[lang] = sum(1 for s in signs if s in code)

        if not scores or max(scores.values()) == 0:
            return "python"
        return max(scores, key=scores.get)


# Singleton
_parser = MultiLanguageParser()


def get_parser() -> MultiLanguageParser:
    return _parser
