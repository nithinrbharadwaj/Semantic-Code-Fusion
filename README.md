# 🧠 Semantic Code Fusion 

> An AI-driven system that merges code from different programming languages, ensuring compatibility and reducing rewrite costs in large-scale software projects.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.3-orange)](https://groq.com)
[![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-red)](https://github.com/facebookresearch/faiss)
[![Celery](https://img.shields.io/badge/Queue-Celery%20%2B%20Redis-purple)](https://docs.celeryq.dev)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🎯 Problem Statement

Large-scale software projects frequently involve code written across multiple programming languages. Manually rewriting or integrating this code is expensive, error-prone, and time-consuming. **Semantic Code Fusion** solves this by using AI to semantically understand and merge code from different languages into a single, compatible, production-ready output.

---

## ✅ Evaluation Results

| Criteria | Implementation | Result |
|---|---|---|
| **Fusion Accuracy** | 6-agent Groq LLM pipeline with cosine similarity scoring | avg similarity **0.52–0.70**, 100% success rate |
| **Retrieval Efficiency** | FAISS IndexFlatL2 + sentence-transformers embeddings | search latency **~130ms** |
| **NLP Integration** | Analyzer → Planner → Fusion → Fixer → Reviewer agents | all 6 agents completing in **8–25 seconds** |
| **API Performance** | FastAPI sync + async endpoints, Celery job queue | 200 OK on all endpoints, async job polling working |
| **Metrics Application** | Cosine similarity, merge success rate, learning report | 5 outcomes tracked, 4 language pairs, 100% success rate |

---

## 🚀 Features

- **Multi-Language Fusion** — Python, JavaScript, TypeScript, Java, Go, Rust
- **6-Agent AI Pipeline** — Analyzer → Planner → Fusion → Fixer → Tester → Reviewer
- **Vector Search** — FAISS-powered semantic code retrieval with local embeddings (no API cost)
- **Async Processing** — Celery + Redis background job queue with real-time polling
- **Conflict Resolution** — Pre-flight detection of naming, signature, and type conflicts
- **Continuous Learning** — Tracks fusion outcomes, learns language-pair patterns over time
- **Code Graph Analysis** — AST-based call graph for intelligent merge ordering
- **Security Scanning** — Static analysis for 15 vulnerability patterns
- **REST API** — FastAPI with Swagger UI at `/docs`
- **Web UI** — Dark/light mode frontend at `http://localhost:8000`

---

## 🏗️ Architecture

```
User Request
     │
     ▼
FastAPI (port 8000)
     │
     ├── Sync  ──► FusionPipeline (6 agents)
     │
     └── Async ──► Celery Task ──► Redis Queue ──► Worker
                                        │
                                        ▼
                               EnhancedFusionPipeline
                               ├── ConflictResolver
                               ├── CodeGraphBuilder
                               ├── 6-Agent LLM (Groq)
                               └── ContinuousLearner
                                        │
                                        ▼
                               PostgreSQL (results)
                               FAISS (vector index)
```

---

## 📁 Project Structure

```
semantic_code_fusion/
├── app/
│   ├── agents/
│   │   ├── pipeline.py           # 6-agent Groq LLM pipeline
│   │   └── enhanced_pipeline.py  # Conflict resolution + learning
│   ├── api/
│   │   ├── middleware.py          # Rate limiting, logging
│   │   └── routes/
│   │       ├── fusion.py          # /fuse, /fuse/async, /migrate
│   │       ├── search.py          # /search, /index
│   │       ├── analyze.py         # /analyze
│   │       ├── jobs.py            # /job/{id}
│   │       ├── metrics.py         # /metrics
│   │       └── advanced.py        # /fuse/enhanced, /conflicts, /graph
│   ├── core/
│   │   ├── conflict_resolver.py   # Pre-fusion conflict detection
│   │   ├── code_graph.py          # AST call graph builder
│   │   ├── learning.py            # Continuous learning system
│   │   ├── database.py            # SQLAlchemy async models
│   │   └── schemas.py             # Pydantic request/response models
│   ├── vector/
│   │   └── store.py               # FAISS vector store (local embeddings)
│   ├── parsers/
│   │   └── ast_parser.py          # Multi-language AST parser
│   ├── utils/
│   │   ├── metrics.py             # Cosine similarity, structural overlap
│   │   ├── security_scanner.py    # Static security analysis
│   │   └── code_utils.py          # Language detection, code helpers
│   ├── tasks.py                   # Celery task definitions
│   ├── worker.py                  # Celery app configuration
│   └── main.py                    # FastAPI app entry point
├── frontend/
│   └── index.html                 # Web UI (dark/light mode)
├── tests/
│   ├── test_fusion_pipeline.py
│   ├── test_api.py
│   └── test_advanced.py
├── alembic/                       # Database migrations
├── scripts/
│   ├── setup.py                   # Automated setup
│   ├── demo.py                    # Live demo script
│   └── cli.py                     # CLI management tool
├── docker/
│   └── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## ⚙️ Setup (Windows)

### Prerequisites
- Python 3.11
- PostgreSQL (port 5432 or 5433)
- Redis (via Docker or local install)
- Groq API key — get free at [console.groq.com](https://console.groq.com)

### 1. Clone & create virtual environment
```powershell
git clone https://github.com/nithinrbharadwaj/Semantic-Code-Fusion.git
cd Semantic-Code-Fusion
python -m venv venv --without-pip
venv\Scripts\Activate.ps1
python -m ensurepip --upgrade
python -m pip install -r requirements.txt
```

### 2. Configure environment
```powershell
copy .env.example .env
```
Open `.env` and set:
```env
OPENAI_API_KEY=gsk_your_groq_key_here
OPENAI_MODEL=llama-3.3-70b-versatile
FAISS_DIMENSION=384
DATABASE_URL=postgresql+asyncpg://scf_user:scf_pass@localhost:5433/semantic_code_fusion
SYNC_DATABASE_URL=postgresql://scf_user:scf_pass@localhost:5433/semantic_code_fusion
```

### 3. Start PostgreSQL & Redis
```powershell
# PostgreSQL — via pgAdmin or psql
# Redis — run from your Redis installation folder
cd C:\Redis
.\redis-server.exe
```

### 4. Run database migrations
```powershell
alembic upgrade head
```

### 5. Start the API server
```powershell
uvicorn app.main:app --reload --port 8000
```

### 6. Start Celery worker (new terminal)
```powershell
venv\Scripts\Activate.ps1
celery -A app.worker worker --loglevel=info --pool=solo
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/fuse` | Sync fusion — merges two code snippets immediately |
| POST | `/api/v1/fuse/async` | Async fusion — enqueues job via Celery |
| POST | `/api/v1/fuse/enhanced` | Enhanced fusion with conflict resolution + learning |
| POST | `/api/v1/fuse/batch` | Batch fusion of multiple code pairs |
| GET | `/api/v1/job/{job_id}` | Poll async job status |
| POST | `/api/v1/search` | Semantic code search via FAISS |
| POST | `/api/v1/index` | Index code snippets for search |
| POST | `/api/v1/analyze` | Static code analysis + security scan |
| POST | `/api/v1/conflicts` | Pre-flight conflict analysis |
| POST | `/api/v1/graph` | Build code dependency graph |
| GET | `/api/v1/metrics` | System performance metrics |
| GET | `/api/v1/learning/report` | Continuous learning report |
| GET | `/docs` | Swagger UI |
| GET | `/` | Web UI |

---

## 🧪 Quick Test

### Fuse Python + JavaScript
```powershell
$body = '{"primary":{"code":"def add(a,b): return a+b","language":"python"},"secondary":{"code":"const sub = (a,b) => a-b;","language":"javascript"},"target_language":"python","strategy":"hybrid","explain":true}'
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/fuse" -Method Post -Body $body -ContentType "application/json"
```

### Index + Search code
```powershell
# Index
$body = @{ snippets = @(@{ code = "def add(a,b): return a+b"; language = "python"; description = "addition function" }) } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/index" -Method Post -Body $body -ContentType "application/json"

# Search
$body = @{ query = "addition function" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/search" -Method Post -Body $body -ContentType "application/json"
```

### Run tests
```powershell
pytest tests/ -v
```

---

## 📊 Metrics Demonstrated

| Metric | Value |
|---|---|
| Languages supported | Python, JavaScript, Java, Go, TypeScript, Rust |
| Avg cosine similarity | 0.52 – 0.70 |
| Merge success rate | 100% (5/5 fusions) |
| Avg pipeline time | 8 – 25 seconds |
| Vector search latency | ~130ms |
| Async jobs tracked | PostgreSQL + Redis |
| Language pairs learned | python+js, java+python, go+js, js+python |
| Continuous learning outcomes | 5 tracked, success rate 1.0 |

---

## 🤖 Agent Pipeline

Each fusion runs through 6 specialized agents powered by **Groq Llama 3.3 70B**:

| Agent | Role | Avg Time |
|---|---|---|
| AnalyzerAgent | Deep semantic analysis of both code snippets | ~2–3s |
| PlannerAgent | Designs the fusion strategy | ~2s |
| FusionAgent | Performs the actual code merge | ~1s |
| FixerAgent | Resolves conflicts, fixes syntax errors | ~1–3s |
| TesterAgent | Auto-generates unit tests (when enabled) | ~2s |
| ReviewerAgent | Produces human-readable explanation | ~1–2s |

---

## 🔒 Security

- 15 vulnerability patterns detected (hardcoded secrets, SQL injection, unsafe deserialization, weak hashing, etc.)
- Per-IP rate limiting (10 fusions/min, 60 general requests/min)
- Input validation via Pydantic schemas
- No API keys stored in code — environment variables only

---

## 📜 License

MIT — free to use, modify, and distribute.