#!/usr/bin/env python3
"""
scripts/setup.py - One-time setup script for Semantic Code Fusion
Fixed for Windows paths with spaces in directory names.
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path


def run_args(args, check=True):
    """Run command as a list — immune to spaces in paths."""
    print(f"  $ {' '.join(str(a) for a in args)}")
    result = subprocess.run([str(a) for a in args], check=check)
    return result.returncode == 0


def run(cmd, check=True):
    """Run simple shell commands that have no user-supplied paths."""
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, check=check)
    return result.returncode == 0


def main():
    print("=" * 60)
    print("  Semantic Code Fusion v2.0 - Setup")
    print("=" * 60)

    # Resolve absolute path so spaces are handled correctly
    root = Path(__file__).resolve().parent.parent
    is_windows = os.name == "nt"

    # ── 1. Python version ────────────────────────────────────────────────
    print("\n[1/6] Checking Python version...")
    if sys.version_info < (3, 10):
        print("ERROR: Python 3.10+ required")
        sys.exit(1)
    print(f"OK    Python {sys.version_info.major}.{sys.version_info.minor}")

    # ── 2. Virtual environment ───────────────────────────────────────────
    print("\n[2/6] Virtual environment...")
    venv_path = root / "venv"
    if not venv_path.exists():
        # List-form avoids every shell quoting / space-in-path problem
        run_args([sys.executable, "-m", "venv", venv_path])
        print("OK    Created venv")
    else:
        print("OK    venv already exists")

    # ── 3. Install dependencies ──────────────────────────────────────────
    print("\n[3/6] Installing dependencies...")

    # Always use the venv's own python to invoke pip — most reliable on Windows
    if is_windows:
        venv_python = venv_path / "Scripts" / "python.exe"
    else:
        venv_python = venv_path / "bin" / "python"

    req_file = root / "requirements.txt"

    print("  Upgrading pip...")
    run_args([venv_python, "-m", "pip", "install", "--upgrade", "pip"])

    print("  Installing requirements (this may take a few minutes)...")
    run_args([venv_python, "-m", "pip", "install", "-r", req_file])

    print("OK    Dependencies installed")

    # ── 4. .env file ─────────────────────────────────────────────────────
    print("\n[4/6] Environment file...")
    env_file     = root / ".env"
    example_file = root / ".env.example"
    if not env_file.exists():
        if example_file.exists():
            shutil.copy(example_file, env_file)
            print("OK    Created .env from .env.example")
        else:
            env_file.write_text("OPENAI_API_KEY=sk-your-key-here\n")
            print("OK    Created blank .env")
        print("  >>> Open .env and set OPENAI_API_KEY=sk-... <<<")
    else:
        print("OK    .env already exists")

    # ── 5. Directories ───────────────────────────────────────────────────
    print("\n[5/6] Creating directories...")
    for d in ["data/faiss_index", "data/learning", "logs", "htmlcov"]:
        (root / d).mkdir(parents=True, exist_ok=True)
    print("OK    Directories created")

    # ── 6. Docker check ──────────────────────────────────────────────────
    print("\n[6/6] Checking Docker...")
    has_docker = run("docker --version", check=False)
    if has_docker:
        run("docker compose version", check=False)   # v2 syntax
        print("OK    Docker available")
    else:
        print("--    Docker not found (optional - see manual steps below)")

    # ── Next steps ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Setup complete!")
    print("=" * 60)

    if is_windows:
        activate_cmd = r"venv\Scripts\activate"
    else:
        activate_cmd = "source venv/bin/activate"

    print(f"""
NEXT STEPS
----------
1. Activate the virtual environment:
   {activate_cmd}

2. Edit .env and paste your OpenAI key:
   OPENAI_API_KEY=sk-...

3. Start PostgreSQL + Redis:
   docker compose up -d postgres redis

4. Run database migrations:
   alembic upgrade head

5. Start the app:
   uvicorn app.main:app --reload --port 8000

6. Open in browser:
   http://localhost:8000/ui      <- Web UI
   http://localhost:8000/docs    <- Swagger API docs
   http://localhost:8000/health  <- Health check

OPTIONAL
--------
- Run demo (tests your API key without a browser):
  python scripts/demo.py

- Run tests:
  pytest tests/ -v

- Start Celery worker (for async /fuse/async jobs):
  celery -A app.celery_app worker --loglevel=info
""")


if __name__ == "__main__":
    main()