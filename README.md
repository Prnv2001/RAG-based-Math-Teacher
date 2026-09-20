# RAG-Based AI Mathematics Teacher (CBSE Class 9 & Class 10)

An intelligent, interactive, RAG-powered Mathematics Teacher web application and API engineered for CBSE Class 9 and Class 10 students. Powered by **FastAPI**, **LangGraph**, **PostgreSQL + pgvector**, **Groq LLM**, **Ollama Embeddings**, **SymPy Symbolic Verification**, and a **Deterministic SVG Geometry Engine**.

---

## 🌟 Key Capabilities

- 📚 **Hybrid Retrieval (Vector + Full-Text Search)**: Searches NCERT Class 9 & 10 textbook chunks using HNSW cosine similarity on `nomic-embed-text` embeddings combined with PostgreSQL keyword search, fused via **Reciprocal Rank Fusion (RRF)**.
- 📐 **Deterministic NCERT Geometry SVG Engine**: Dynamically generates 100% geometrically accurate inline vector diagrams for Triangles, Circles (Class 9 Chords, Cyclic Quadrilaterals, Class 10 Tangents), Right Triangles (90° corner boxes), and Cone/3D shapes with anti-overlap label positioning.
- 🧮 **SymPy Symbolic Math Verifier**: Algebraically verifies proposed polynomial roots and equation solutions before outputting results to guarantee mathematical correctness.
- 🔄 **Context-Aware Query Expansion**: Automatically resolves follow-up action chips (`Show the full worked solution`, `Need a Hint`, `Option A: 4x² - 3x + 1`) by merging with prior chat context to maintain RAG grounding.
- 🚀 **Pedagogical Progression & Real-World Analogies**: Explains complex topics using real-world analogies (recipe ingredients for polynomials, phone battery for linear equations, basketball arcs for parabolas) and smoothly progresses through chapter roadmaps without looping quizzes endlessly.
- 📄 **PDF Exam Generator**: Conducts customized practice tests and exports clean, printable PDF question papers with optional solution keys.
- 💬 **Embeddable Chat Widget (`chat-widget.js`)**: Single-file JavaScript widget that embeds an interactive dark-mode AI Math Tutor into any LMS or web application with floating toggle button and PDF export.

---

## 🛠️ Architecture & Technology Stack

| Layer | Technology |
| :--- | :--- |
| **Backend Framework** | Python 3.10+, FastAPI, Uvicorn, LangGraph |
| **Vector Database** | PostgreSQL 15+ with `pgvector` HNSW index |
| **Embeddings** | Ollama (`nomic-embed-text`, 768 dimensions) |
| **LLM Inference** | Groq Cloud API (`openai/gpt-oss-20b` / `llama-3.3-70b-versatile`) |
| **Symbolic Verification** | SymPy (Deterministic equation & root verification) |
| **Diagram Engine** | Custom SVG Vector Engine (`svg_engine.py`) |
| **Database ORM & Migrations** | SQLAlchemy 2.0 (Async), Alembic |
| **Frontend Widget** | Vanilla JS (`chat-widget.js`), CSS Glassmorphism, HTML5 |

---

## 📋 Prerequisites & Requirements

1. **Python 3.10 or higher**
2. **PostgreSQL 15+** with the `pgvector` extension installed (`CREATE EXTENSION IF NOT EXISTS vector;`)
3. **Ollama** running locally on port `11434` with the embedding model pulled:
   ```bash
   ollama pull nomic-embed-text
   ```
4. **Groq API Key** (Get a free API key at [console.groq.com](https://console.groq.com))

---

## 🚀 Quick Start & Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/Prnv2001/RAG-based-Math-Teacher.git
cd RAG-based-Math-Teacher
```

### Step 2: Create & Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -e .
```

### Step 4: Configure Environment Variables
Copy the `.env.example` template to `.env` and fill in your Database URL and Groq API Key:
```bash
cp .env.example .env
```

Edit `.env`:
```env
# Database
DATABASE_URL=postgresql+asyncpg://postgres:your_password@localhost:5432/math_teacher

# Groq LLM API
GROQ_API_KEY=gsk_your_primary_groq_key_here
GROQ_API_KEY_FALLBACK=gsk_your_fallback_groq_key_here
GROQ_MODEL=openai/gpt-oss-20b

# Ollama Embeddings
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSIONS=768

# RAG Quality
GROUNDING_THRESHOLD=0.80
MAX_AGENT_RETRIES=2

# Security Key for Ingestion
ADMIN_API_KEY=your-secure-admin-key
```

### Step 5: Database Setup & Migrations
Ensure PostgreSQL is running and the database exists, then run Alembic migrations to create tables and vector indexes:
```bash
alembic upgrade head
```

### Step 6: Ingest Textbook Curriculum Data (Vector DB Setup)
Populate the vector database with official NCERT Class 9 and Class 10 Mathematics textbook chunks:
```bash
# Run the database ingestion script
python -m math_teacher.rag.ingest
```
*Alternatively, call the ingestion API endpoint using your `ADMIN_API_KEY`:*
```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
     -H "X-Admin-Key: your-secure-admin-key"
```

### Step 7: Launch the Development Server
```bash
python -m uvicorn math_teacher.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 🌐 Embeddable Chat Widget Integration

You can easily embed the AI Math Teacher chat widget into any webpage or LMS by including `chat-widget.js` and calling `initMathTeacherWidget()`.

### Widget HTML Embed Script Code
Add the following code snippet before the closing `</body>` tag of your HTML page:

```html
<!-- AI Math Teacher Chat Widget -->
<script src="http://localhost:8000/chat-widget.js"></script>
<script>
  document.addEventListener('DOMContentLoaded', function() {
    window.initMathTeacherWidget({
      apiUrl: "http://localhost:8000/api/v1/chat",
      classLevel: 9,              // Default class level (9 or 10)
      theme: "dark",               // UI Theme ("dark" or "light")
      position: "bottom-right",    // Floating button position
      title: "AI Math Teacher 🌟"  // Header title
    });
  });
</script>
```

---

## 🔗 Important URLs & API Endpoints

Once the Uvicorn server is running on `http://localhost:8000`, access the following URLs:

### Interactive UI Pages & Documentation
- 🌐 **Interactive Web Chat Interface**: [http://localhost:8000/chat.html](http://localhost:8000/chat.html)
- 📖 **Swagger UI API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- 📜 **Redoc API Specification**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- 💚 **Health Check Endpoint**: [http://localhost:8000/health](http://localhost:8000/health)

### Core REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/chat` | Main conversational pipeline (orchestrates router, retrieval, tutor/solver/exam agents, SymPy verifier, and grounding guardrails). |
| `POST` | `/api/v1/ingest` | Admin endpoint to ingest textbook documents into pgvector (Requires `X-Admin-Key` header). |
| `POST` | `/api/v1/exam/pdf` | Generates a downloadable PDF file for an exam paper. |
| `GET` | `/health` | Server and database status check. |

#### Sample Request (`POST /api/v1/chat`):
```json
{
  "question": "Which of the following is a polynomial? A) 3/x + 4  B) 7x³ - 2x + 1  C) √x + 5  D) 1/x² + 3",
  "class_level": 9,
  "chapter": "Polynomials",
  "mode": "INTERACTIVE",
  "chat_history": []
}
```

---

## 🧪 Running Unit Tests

Run the complete automated unit test suite to verify RAG retrieval, SymPy verifier, text sanitizer, and SVG diagram engines:

```bash
pytest tests/unit/
```

*Expected output:*
```text
============================= 42 passed in 1.37s ==============================
```

---

## 📁 Repository Structure

```text
math_teacher_ai/
├── src/math_teacher/
│   ├── agents/            # Router, Tutor, Solver, and Exam Agents
│   ├── config/            # Settings, Prompts (Tutor v2, Router v1), Environment
│   ├── domain/            # Models, Enums (SolverMode, Intent)
│   ├── geometry/          # SVG Vector Engine (GeometrySVGBuilder)
│   ├── guardrails/        # Input and Output Grounding Checkers
│   ├── math/              # SymPy MathVerifier
│   ├── orchestration/     # LangGraph Nodes & Pipeline Execution
│   ├── rag/               # Vector Search, Keyword Search, RRF, Context Builder
│   ├── storage/           # SQLAlchemy DB Models & Alembic Setup
│   ├── utils/             # Text Sanitizer & PDF Generator
│   └── main.py            # FastAPI Application Entry Point
├── tests/unit/            # Automated Unit Test Suite (42 tests)
├── migrations/            # Alembic DB Migration Scripts
├── chat-widget.js         # Embeddable Frontend Chat Widget Script
├── chat.html              # Standalone Chat Interface Test Page
├── .env.example           # Environment Configuration Template
├── pyproject.toml         # Dependencies & Build Specification
└── README.md              # Project Documentation
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.