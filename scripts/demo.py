#!/usr/bin/env python3
"""
scripts/demo.py - Live demonstration of Semantic Code Fusion

Runs the fusion pipeline directly (no HTTP) for quick testing.
Requires OPENAI_API_KEY in environment.
"""
import asyncio
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from rich.table import Table

console = Console()

# ─── Demo Code Pairs ──────────────────────────────────────────────────────────

DEMOS = {
    "py_js": {
        "title": "Python Math + JavaScript DataProcessor",
        "primary": {
            "code": '''import math
from typing import List, Optional

def calculate_statistics(numbers: List[float]) -> dict:
    """Calculate basic statistics for a list of numbers."""
    if not numbers:
        return {}
    n = len(numbers)
    mean = sum(numbers) / n
    variance = sum((x - mean) ** 2 for x in numbers) / n
    return {
        "count": n,
        "mean": round(mean, 4),
        "std_dev": round(math.sqrt(variance), 4),
        "min": min(numbers),
        "max": max(numbers),
        "median": sorted(numbers)[n // 2],
    }

class DataValidator:
    def __init__(self, rules: dict):
        self.rules = rules

    def validate(self, data: dict) -> tuple[bool, list]:
        errors = []
        for field, rule in self.rules.items():
            if rule.get("required") and field not in data:
                errors.append(f"Missing required field: {field}")
            if field in data and "type" in rule:
                if not isinstance(data[field], rule["type"]):
                    errors.append(f"Invalid type for {field}")
        return len(errors) == 0, errors
''',
            "language": "python",
        },
        "secondary": {
            "code": '''class AsyncDataFetcher {
    constructor(baseUrl, options = {}) {
        this.baseUrl = baseUrl;
        this.timeout = options.timeout || 5000;
        this.retries = options.retries || 3;
        this.cache = new Map();
    }

    async fetch(endpoint, useCache = true) {
        const url = `${this.baseUrl}${endpoint}`;
        if (useCache && this.cache.has(url)) {
            return { data: this.cache.get(url), fromCache: true };
        }
        for (let attempt = 1; attempt <= this.retries; attempt++) {
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), this.timeout);
                const response = await fetch(url, { signal: controller.signal });
                clearTimeout(timeoutId);
                const data = await response.json();
                this.cache.set(url, data);
                return { data, fromCache: false, attempt };
            } catch (err) {
                if (attempt === this.retries) throw err;
                await new Promise(r => setTimeout(r, attempt * 1000));
            }
        }
    }

    clearCache() { this.cache.clear(); }
}
''',
            "language": "javascript",
        },
        "target_language": "python",
    }
}


async def run_demo():
    from app.agents.pipeline import FusionPipeline
    from app.core.schemas import FusionRequest, CodeSnippet, Language, FusionStrategy

    demo = DEMOS["py_js"]
    console.print(Panel(f"[bold cyan]Demo: {demo['title']}[/bold cyan]", border_style="cyan"))

    request = FusionRequest(
        primary=CodeSnippet(
            code=demo["primary"]["code"],
            language=Language(demo["primary"]["language"]),
        ),
        secondary=CodeSnippet(
            code=demo["secondary"]["code"],
            language=Language(demo["secondary"]["language"]),
        ),
        target_language=Language(demo["target_language"]),
        strategy=FusionStrategy.HYBRID,
        explain=True,
        run_tests=True,
    )

    console.print("\n[yellow]Running 6-agent pipeline...[/yellow]\n")

    pipeline = FusionPipeline()

    try:
        result = await pipeline.run(request, "demo-001")
    except Exception as e:
        console.print(f"[red]Pipeline error: {e}[/red]")
        console.print("[yellow]Hint: Make sure OPENAI_API_KEY is set in your .env[/yellow]")
        return

    # Metrics table
    if result.metrics:
        m = result.metrics
        table = Table(title="Fusion Metrics", show_header=True, header_style="cyan")
        table.add_column("Metric")
        table.add_column("Value", style="green")
        table.add_row("Cosine Similarity", f"{m.cosine_similarity:.4f}")
        table.add_row("Structural Overlap", f"{m.structural_overlap:.4f}")
        table.add_row("Merge Success Rate", f"{m.merge_success_rate:.1%}")
        table.add_row("Processing Time", f"{m.processing_time_ms:.0f} ms")
        table.add_row("Tokens Used", str(m.tokens_used))
        table.add_row("Lines Added", str(m.lines_added))
        console.print(table)

    # Agent traces
    console.print("\n[bold]Agent Pipeline Traces:[/bold]")
    for trace in result.agent_traces:
        console.print(f"  [cyan]{trace.agent}[/cyan] [{trace.duration_ms:.0f}ms] → {trace.result[:80]}...")

    # Explanation
    if result.explanation:
        console.print(Panel(result.explanation, title="AI Explanation", border_style="blue"))

    # Fused code
    if result.fused_code:
        console.print("\n[bold]Fused Code:[/bold]")
        console.print(Syntax(result.fused_code, "python", theme="monokai", line_numbers=True))

    # Tests
    if result.test_cases:
        console.print("\n[bold]Generated Tests:[/bold]")
        console.print(Syntax(result.test_cases[:1000], "python", theme="monokai"))

    # Warnings
    for w in result.warnings:
        console.print(f"[yellow]⚠️  {w}[/yellow]")

    console.print(f"\n[green]✅ Demo complete! Status: {result.status.value}[/green]")


if __name__ == "__main__":
    # Load .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]❌ OPENAI_API_KEY not set. Add it to .env[/red]")
        sys.exit(1)

    asyncio.run(run_demo())
