#!/usr/bin/env python3
"""
scripts/cli.py - Management CLI for Semantic Code Fusion

Usage:
    python scripts/cli.py fuse --primary file1.py --secondary file2.js --target python
    python scripts/cli.py search "fibonacci algorithm"
    python scripts/cli.py analyze mycode.py
    python scripts/cli.py index --dir ./my_codebase
    python scripts/cli.py metrics
"""
import asyncio
import json
import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import httpx

app = typer.Typer(name="scf", help="Semantic Code Fusion CLI")
console = Console()
BASE_URL = os.getenv("SCF_API_URL", "http://localhost:8000")


# ─── Fuse Command ─────────────────────────────────────────────────────────────

@app.command()
def fuse(
    primary: Path = typer.Option(..., "--primary", "-p", help="Primary source file"),
    secondary: Path = typer.Option(..., "--secondary", "-s", help="Secondary source file"),
    target: str = typer.Option("python", "--target", "-t", help="Target language"),
    strategy: str = typer.Option("hybrid", "--strategy", help="Fusion strategy"),
    output: Path = typer.Option(None, "--output", "-o", help="Save output to file"),
    explain: bool = typer.Option(True, "--explain/--no-explain", help="Include explanation"),
    tests: bool = typer.Option(False, "--tests", help="Generate test cases"),
):
    """Fuse two code files using AI."""
    if not primary.exists():
        console.print(f"[red]File not found: {primary}[/red]")
        raise typer.Exit(1)
    if not secondary.exists():
        console.print(f"[red]File not found: {secondary}[/red]")
        raise typer.Exit(1)

    primary_code = primary.read_text()
    secondary_code = secondary.read_text()

    p_lang = _detect_lang_from_ext(primary.suffix)
    s_lang = _detect_lang_from_ext(secondary.suffix)

    payload = {
        "primary": {"code": primary_code, "language": p_lang},
        "secondary": {"code": secondary_code, "language": s_lang},
        "target_language": target,
        "strategy": strategy,
        "explain": explain,
        "run_tests": tests,
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Running fusion pipeline..."),
        transient=True,
    ) as progress:
        progress.add_task("fuse", total=None)
        try:
            response = httpx.post(f"{BASE_URL}/api/v1/fuse", json=payload, timeout=180)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            console.print(f"[red]API error: {e.response.text}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

    data = response.json()

    # Display result
    console.print(Panel(
        f"[green]✅ Fusion Complete[/green]\n"
        f"Job ID: [cyan]{data['job_id']}[/cyan]\n"
        f"Status: [green]{data['status']}[/green]",
        title="Semantic Code Fusion",
    ))

    if data.get("metrics"):
        m = data["metrics"]
        table = Table(title="Fusion Metrics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Cosine Similarity", f"{m['cosine_similarity']:.4f}")
        table.add_row("Structural Overlap", f"{m['structural_overlap']:.4f}")
        table.add_row("Processing Time", f"{m['processing_time_ms']:.0f} ms")
        table.add_row("Tokens Used", str(m['tokens_used']))
        table.add_row("Lines Added", str(m['lines_added']))
        console.print(table)

    if data.get("explanation"):
        console.print(Panel(data["explanation"], title="[bold]AI Explanation[/bold]", border_style="blue"))

    if data.get("fused_code"):
        console.print("\n[bold]Fused Code:[/bold]")
        syntax = Syntax(data["fused_code"], target, theme="monokai", line_numbers=True)
        console.print(syntax)

        if output:
            output.write_text(data["fused_code"])
            console.print(f"\n[green]✅ Saved to {output}[/green]")

    if data.get("warnings"):
        for w in data["warnings"]:
            console.print(f"[yellow]⚠️  {w}[/yellow]")


# ─── Search Command ───────────────────────────────────────────────────────────

@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
    language: str = typer.Option(None, "--language", "-l", help="Filter by language"),
    min_sim: float = typer.Option(0.5, "--min-sim", help="Minimum similarity score"),
):
    """Search indexed code snippets semantically."""
    payload = {"query": query, "top_k": top_k, "min_similarity": min_sim}
    if language:
        payload["language"] = language

    try:
        response = httpx.post(f"{BASE_URL}/api/v1/search", json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Found {data['total']} results in {data['search_time_ms']:.1f}ms[/bold]\n")
    for i, result in enumerate(data["results"], 1):
        console.print(f"[bold cyan]#{i}[/bold cyan] | Language: [green]{result['language']}[/green] | Similarity: [yellow]{result['similarity']:.4f}[/yellow]")
        if result.get("description"):
            console.print(f"  {result['description']}")
        syntax = Syntax(result["code"][:300] + ("..." if len(result["code"]) > 300 else ""),
                        result["language"], theme="monokai")
        console.print(syntax)
        console.print()


# ─── Analyze Command ──────────────────────────────────────────────────────────

@app.command()
def analyze(
    file: Path = typer.Argument(..., help="Source code file to analyze"),
    language: str = typer.Option("auto", "--language", "-l"),
):
    """Analyze code quality, complexity, and security."""
    if not file.exists():
        console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)

    code = file.read_text()
    payload = {"code": code, "language": language}

    try:
        response = httpx.post(f"{BASE_URL}/api/v1/analyze", json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    m = data["metrics"]
    table = Table(title=f"Code Quality Report: {file.name}", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Language", data["language"])
    table.add_row("Lines of Code", str(m["lines_of_code"]))
    table.add_row("Cyclomatic Complexity", str(m["cyclomatic_complexity"]))
    table.add_row("Comment Ratio", f"{m['comment_ratio']:.1%}")
    table.add_row("Duplication Score", f"{m['duplication_score']:.1%}")
    table.add_row("Overall Score", f"{m['overall_score']}/100")
    console.print(table)

    if m.get("security_issues"):
        console.print(f"\n[bold red]Security Issues ({len(m['security_issues'])}):[/bold red]")
        for issue in m["security_issues"]:
            color = {"high": "red", "medium": "yellow", "low": "blue"}[issue["severity"]]
            console.print(f"  [{color}][{issue['severity'].upper()}][/{color}] Line {issue['line']}: {issue['issue']}")
            console.print(f"    → {issue['recommendation']}")

    if data.get("suggestions"):
        console.print("\n[bold]Suggestions:[/bold]")
        for s in data["suggestions"]:
            console.print(f"  • {s}")


# ─── Index Command ────────────────────────────────────────────────────────────

@app.command()
def index(
    directory: Path = typer.Option(..., "--dir", "-d", help="Directory to index"),
    namespace: str = typer.Option("default", "--namespace", "-n"),
    extensions: str = typer.Option(".py,.js,.ts,.java,.go,.rs", "--ext"),
):
    """Index a codebase directory for semantic search."""
    exts = extensions.split(",")
    snippets = []

    for ext in exts:
        for f in directory.rglob(f"*{ext}"):
            try:
                code = f.read_text(encoding="utf-8", errors="ignore")
                if 10 < len(code) < 50000:
                    lang = _detect_lang_from_ext(ext)
                    snippets.append({
                        "code": code,
                        "language": lang,
                        "description": str(f.relative_to(directory)),
                        "metadata": {"file": str(f)},
                    })
            except Exception:
                pass

    if not snippets:
        console.print("[yellow]No eligible files found.[/yellow]")
        raise typer.Exit(0)

    console.print(f"Found [cyan]{len(snippets)}[/cyan] files. Indexing...")

    # Batch into chunks of 50
    batch_size = 50
    indexed = 0
    for i in range(0, len(snippets), batch_size):
        batch = snippets[i:i + batch_size]
        try:
            response = httpx.post(
                f"{BASE_URL}/api/v1/index",
                json={"snippets": batch, "namespace": namespace},
                timeout=120,
            )
            response.raise_for_status()
            indexed += len(batch)
            console.print(f"  Indexed {indexed}/{len(snippets)}...")
        except Exception as e:
            console.print(f"[red]Batch error: {e}[/red]")

    console.print(f"[green]✅ Indexed {indexed} files into namespace '{namespace}'[/green]")


# ─── Metrics Command ──────────────────────────────────────────────────────────

@app.command()
def metrics():
    """Display system performance metrics."""
    try:
        response = httpx.get(f"{BASE_URL}/api/v1/metrics", timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    table = Table(title="System Metrics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total Fusions", str(data["total_fusions"]))
    table.add_row("Successful", str(data["successful_fusions"]))
    table.add_row("Success Rate", f"{data['success_rate']:.1%}")
    table.add_row("Avg Processing Time", f"{data['avg_processing_time_ms']:.0f} ms")
    table.add_row("Avg Cosine Similarity", f"{data['avg_cosine_similarity']:.4f}")
    table.add_row("Indexed Snippets", str(data["total_indexed_snippets"]))
    table.add_row("Uptime", f"{data['uptime_seconds']:.0f}s")
    table.add_row("Supported Languages", ", ".join(data["supported_languages"]))
    console.print(table)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _detect_lang_from_ext(ext: str) -> str:
    mapping = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".java": "java", ".go": "go", ".rs": "rust", ".cpp": "cpp",
        ".cs": "csharp",
    }
    return mapping.get(ext.lower(), "python")


if __name__ == "__main__":
    app()
