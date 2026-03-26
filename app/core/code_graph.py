"""
app/core/code_graph.py - Code dependency and call graph builder

Builds a directed graph of function calls, class hierarchies,
and module dependencies. Used for intelligent fusion ordering
and visualization.
"""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple


@dataclass
class GraphNode:
    id: str
    name: str
    node_type: str          # function | class | module | variable
    language: str
    start_line: int
    end_line: int
    snippet: str = ""       # First 150 chars
    calls: List[str] = field(default_factory=list)      # Functions this calls
    called_by: List[str] = field(default_factory=list)  # Functions that call this
    imports: List[str] = field(default_factory=list)


@dataclass
class CodeGraph:
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    edges: List[Tuple[str, str, str]] = field(default_factory=list)  # (from, to, edge_type)
    language: str = "unknown"

    def add_node(self, node: GraphNode):
        self.nodes[node.id] = node

    def add_edge(self, from_id: str, to_id: str, edge_type: str = "calls"):
        self.edges.append((from_id, to_id, edge_type))

    def get_entry_points(self) -> List[GraphNode]:
        """Nodes not called by anyone — potential entry points."""
        called_ids = {to for _, to, _ in self.edges}
        return [n for nid, n in self.nodes.items() if nid not in called_ids]

    def get_call_order(self) -> List[str]:
        """Topological sort for fusion ordering (leaves first)."""
        in_degree: Dict[str, int] = {nid: 0 for nid in self.nodes}
        for _, to, _ in self.edges:
            if to in in_degree:
                in_degree[to] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order = []
        while queue:
            node_id = queue.pop(0)
            order.append(node_id)
            for from_id, to_id, _ in self.edges:
                if from_id == node_id:
                    in_degree[to_id] -= 1
                    if in_degree[to_id] == 0:
                        queue.append(to_id)

        return order

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "nodes": [
                {
                    "id": n.id,
                    "name": n.name,
                    "type": n.node_type,
                    "line": n.start_line,
                    "calls": n.calls,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {"from": f, "to": t, "type": et}
                for f, t, et in self.edges
            ],
        }


class CodeGraphBuilder:
    """Builds call/dependency graphs from source code."""

    def build(self, code: str, language: str) -> CodeGraph:
        graph = CodeGraph(language=language)
        lines = code.split("\n")

        if language.lower() == "python":
            self._build_python_graph(code, lines, graph)
        elif language.lower() in ("javascript", "typescript"):
            self._build_js_graph(code, lines, graph)
        elif language.lower() == "java":
            self._build_java_graph(code, lines, graph)
        elif language.lower() == "go":
            self._build_go_graph(code, lines, graph)
        else:
            self._build_generic_graph(code, lines, graph)

        return graph

    def _build_python_graph(self, code: str, lines: List[str], graph: CodeGraph):
        """Parse Python into graph."""
        # Extract classes
        class_pattern = re.compile(r"^class\s+(\w+)(?:\(([^)]*)\))?:", re.MULTILINE)
        for m in class_pattern.finditer(code):
            name = m.group(1)
            bases = [b.strip() for b in (m.group(2) or "").split(",") if b.strip()]
            line_num = code[:m.start()].count("\n") + 1
            node = GraphNode(
                id=f"class:{name}",
                name=name,
                node_type="class",
                language="python",
                start_line=line_num,
                end_line=line_num + 20,
                snippet=m.group(0)[:100],
            )
            graph.add_node(node)
            # Inheritance edges
            for base in bases:
                if base not in ("object", "ABC", "Exception", "BaseException"):
                    graph.add_edge(f"class:{name}", f"class:{base}", "inherits")

        # Extract functions/methods
        func_pattern = re.compile(
            r"^(?P<indent>\s*)(?:async\s+)?def\s+(?P<name>\w+)\s*\((?P<params>[^)]*)\)",
            re.MULTILINE,
        )
        func_bodies: Dict[str, str] = {}
        func_positions = []

        for m in func_pattern.finditer(code):
            name = m.group("name")
            line_num = code[:m.start()].count("\n") + 1
            indent = len(m.group("indent"))

            # Find body
            body_start = m.end()
            body_lines = []
            for lnum, line in enumerate(lines[line_num:], line_num + 1):
                stripped = line.lstrip()
                if stripped and len(line) - len(stripped) <= indent and lnum > line_num:
                    break
                body_lines.append(line)

            body = "\n".join(body_lines[:30])
            func_bodies[name] = body

            node = GraphNode(
                id=f"func:{name}",
                name=name,
                node_type="method" if indent > 0 else "function",
                language="python",
                start_line=line_num,
                end_line=line_num + len(body_lines),
                snippet=body[:150],
            )
            graph.add_node(node)
            func_positions.append((name, line_num, indent))

        # Build call edges
        all_func_names = set(func_bodies.keys())
        for caller_name, body in func_bodies.items():
            # Find all function calls in body
            call_pattern = re.compile(r"\b(\w+)\s*\(")
            for cm in call_pattern.finditer(body):
                callee = cm.group(1)
                if callee in all_func_names and callee != caller_name:
                    graph.add_edge(f"func:{caller_name}", f"func:{callee}", "calls")
                    if f"func:{caller_name}" in graph.nodes:
                        graph.nodes[f"func:{caller_name}"].calls.append(callee)
                    if f"func:{callee}" in graph.nodes:
                        graph.nodes[f"func:{callee}"].called_by.append(caller_name)

        # Extract imports
        import_pattern = re.compile(r"^(?:import|from)\s+(\S+)", re.MULTILINE)
        for m in import_pattern.finditer(code):
            module = m.group(1).split(".")[0]
            node = GraphNode(
                id=f"module:{module}",
                name=module,
                node_type="module",
                language="python",
                start_line=code[:m.start()].count("\n") + 1,
                end_line=code[:m.start()].count("\n") + 1,
            )
            graph.add_node(node)

    def _build_js_graph(self, code: str, lines: List[str], graph: CodeGraph):
        """Parse JavaScript/TypeScript into graph."""
        # Arrow functions + regular functions
        patterns = [
            re.compile(r"(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)"),
            re.compile(r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|\w+)\s*=>"),
            re.compile(r"(\w+)\s*\(([^)]*)\)\s*\{"),  # Method shorthand
        ]
        seen = set()
        for pattern in patterns:
            for m in pattern.finditer(code):
                name = m.group(1)
                if name in seen or name in ("if", "for", "while", "switch"):
                    continue
                seen.add(name)
                line_num = code[:m.start()].count("\n") + 1
                graph.add_node(GraphNode(
                    id=f"func:{name}", name=name, node_type="function",
                    language="javascript", start_line=line_num, end_line=line_num + 15,
                    snippet=code[m.start():m.start() + 150],
                ))

        # Classes
        for m in re.finditer(r"class\s+(\w+)(?:\s+extends\s+(\w+))?", code):
            name, base = m.group(1), m.group(2)
            line_num = code[:m.start()].count("\n") + 1
            graph.add_node(GraphNode(
                id=f"class:{name}", name=name, node_type="class",
                language="javascript", start_line=line_num, end_line=line_num + 30,
            ))
            if base:
                graph.add_edge(f"class:{name}", f"class:{base}", "extends")

    def _build_java_graph(self, code: str, lines: List[str], graph: CodeGraph):
        """Parse Java into graph."""
        for m in re.finditer(
            r"(?:public|private|protected|static|\s)+[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+\w+)?\s*\{",
            code
        ):
            name = m.group(1)
            if name in ("if", "for", "while", "switch", "catch"):
                continue
            line_num = code[:m.start()].count("\n") + 1
            graph.add_node(GraphNode(
                id=f"func:{name}", name=name, node_type="method",
                language="java", start_line=line_num, end_line=line_num + 20,
            ))

        for m in re.finditer(r"(?:public|private|abstract|final|\s)*class\s+(\w+)(?:\s+extends\s+(\w+))?", code):
            name, base = m.group(1), m.group(2)
            line_num = code[:m.start()].count("\n") + 1
            graph.add_node(GraphNode(
                id=f"class:{name}", name=name, node_type="class",
                language="java", start_line=line_num, end_line=line_num + 50,
            ))
            if base:
                graph.add_edge(f"class:{name}", f"class:{base}", "extends")

    def _build_go_graph(self, code: str, lines: List[str], graph: CodeGraph):
        """Parse Go into graph."""
        for m in re.finditer(r"func\s+(?:\(\w+\s+\*?(\w+)\)\s+)?(\w+)\s*\(", code):
            receiver, name = m.group(1), m.group(2)
            line_num = code[:m.start()].count("\n") + 1
            graph.add_node(GraphNode(
                id=f"func:{name}", name=name,
                node_type="method" if receiver else "function",
                language="go", start_line=line_num, end_line=line_num + 20,
            ))

        for m in re.finditer(r"type\s+(\w+)\s+struct", code):
            name = m.group(1)
            line_num = code[:m.start()].count("\n") + 1
            graph.add_node(GraphNode(
                id=f"struct:{name}", name=name, node_type="class",
                language="go", start_line=line_num, end_line=line_num + 15,
            ))

    def _build_generic_graph(self, code: str, lines: List[str], graph: CodeGraph):
        """Fallback: extract any function-like constructs."""
        for m in re.finditer(r"\b(\w+)\s*\(", code):
            name = m.group(1)
            if len(name) < 3 or name[0].isdigit():
                continue
            if name not in graph.nodes:
                line_num = code[:m.start()].count("\n") + 1
                graph.add_node(GraphNode(
                    id=f"func:{name}", name=name, node_type="function",
                    language="unknown", start_line=line_num, end_line=line_num + 5,
                ))

    def merge_graphs(self, graph_a: CodeGraph, graph_b: CodeGraph) -> CodeGraph:
        """
        Merge two graphs from different languages into a combined view.
        Used for visualizing cross-language dependencies.
        """
        merged = CodeGraph(language=f"{graph_a.language}+{graph_b.language}")

        # Add all nodes with source tag
        for nid, node in graph_a.nodes.items():
            node_copy = GraphNode(**node.__dict__)
            node_copy.id = f"primary:{nid}"
            merged.add_node(node_copy)

        for nid, node in graph_b.nodes.items():
            node_copy = GraphNode(**node.__dict__)
            node_copy.id = f"secondary:{nid}"
            merged.add_node(node_copy)

        # Add edges with source prefix
        for f, t, et in graph_a.edges:
            merged.add_edge(f"primary:{f}", f"primary:{t}", et)
        for f, t, et in graph_b.edges:
            merged.add_edge(f"secondary:{f}", f"secondary:{t}", et)

        # Add cross-language edges for matching names
        primary_names = {n.name: nid for nid, n in graph_a.nodes.items()}
        secondary_names = {n.name: nid for nid, n in graph_b.nodes.items()}
        shared_names = set(primary_names.keys()) & set(secondary_names.keys())

        for name in shared_names:
            merged.add_edge(
                f"primary:{primary_names[name]}",
                f"secondary:{secondary_names[name]}",
                "semantic_match",
            )

        return merged
