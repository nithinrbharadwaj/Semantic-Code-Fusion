# 🧠 Semantic Code Fusion v2.0

An AI-powered system that merges code from different programming languages using semantic understanding, vector search, and multi-agent LLMs.

## 🚀 Features

- **Multi-language Support**: Python, JavaScript, Java, Go, TypeScript, Rust
- **AST-based Parsing**: Tree-sitter for deep code structure analysis
- **Vector Search**: FAISS-powered semantic code retrieval
- **Multi-Agent AI Pipeline**: Analyzer → Fusion → Fixer → Tester agents
- **Conflict Resolution Engine**: Intelligent merge conflict handling
- **Code Quality Analysis**: Security scanning + quality metrics
- **REST API**: FastAPI with async support
- **Background Jobs**: Celery + Redis task queue
- **Auto Test Generation**: AI-generated test cases for merged code
- **Explainable AI**: Human-readable fusion rationale

## 📁 Project Structure

```
semantic_code_fusion/
├── app/
│   ├── api/            # FastAPI routes
│   ├── agents/         # Multi-agent LLM system
│   ├── core/           # Core fusion engine
│   ├── parsers/        # AST parsers per language
│   ├── vector/         # FAISS vector store
│   └── utils/          # Helpers, metrics, security
├── tests/              # Unit + integration tests
├── docker/             # Docker configs
├── scripts/            # Setup & utility scripts
├── frontend/           # Simple web UI
├── .env.example
├── requirements.txt
├── docker-compose.yml
└── README.md
```

## ⚙️ Setup

### 1. Clone & Install
```bash
git clone <repo>
cd semantic_code_fusion
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Start Services
```bash
docker-compose up -d  # Start Redis + PostgreSQL
```

### 4. Run the App
```bash
uvicorn app.main:app --reload --port 8000
```

### 5. Start Celery Worker
```bash
celery -A app.celery_app worker --loglevel=info
```

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/fuse` | Fuse two code snippets |
| POST | `/api/v1/fuse/async` | Async fusion job |
| GET | `/api/v1/job/{job_id}` | Check job status |
| POST | `/api/v1/search` | Semantic code search |
| POST | `/api/v1/analyze` | Analyze code quality |
| POST | `/api/v1/migrate` | Language migration |
| GET | `/api/v1/metrics` | System metrics |
| GET | `/docs` | Swagger UI |

## 🧪 Run Tests
```bash
pytest tests/ -v --cov=app
```

## 📊 Metrics

- **Fusion Accuracy**: Cosine similarity score between source and fused output
- **Retrieval Efficiency**: Vector search latency (ms)
- **Merge Success Rate**: % of successful fusions
- **Code Quality Score**: Complexity, duplication, security flags

## 🐳 Docker Deployment
```bash
docker-compose up --build
```
Access at `http://localhost:8000`

## 📜 License
MIT
