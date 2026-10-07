# Agentic RAG & Multi-Agent Q&A Platform

A job-description-driven Agentic AI project built around **Python, LangChain, LangGraph, LLMs, RAG, tool calling, SQL, FastAPI, evaluation, and observability**.

## Architecture

User -> Supervisor Agent -> RAG Agent / SQL Agent / Data Agent / Calculator -> Response Agent -> Validation Agent -> Final Answer

### Core capabilities
- **Agentic orchestration:** LangGraph stateful workflow with conditional routing.
- **LangChain tools:** typed tool wrappers for retrieval, SQL, Pandas analysis, and arithmetic.
- **RAG:** TF-IDF vector retrieval over a local document collection with top-k evidence.
- **Structured data:** read-only SQLite analytics database and SQL tool.
- **Data analysis:** Pandas-based analysis tool.
- **LLM integration:** optional Anthropic API for supervisor routing and grounded response generation.
- **Backend:** FastAPI `/api/query` endpoint and `/health` endpoint.
- **Validation:** evidence-grounding checks before returning a response.
- **Evaluation:** route accuracy, task success, validation status, step count, and latency.

## JD mapping

| Job requirement | Implementation |
|---|---|
| Python | Core application and tools |
| LangChain | `langchain_core.tools` tool layer |
| LangGraph | Supervisor + specialist-agent orchestration |
| Agentic AI | Conditional routing and tool selection |
| LLMs | Anthropic integration |
| RAG | Vectorized document retrieval + grounded generation |
| Multi-agent workflows | Supervisor, RAG, SQL, Data, Validation agents |
| API integration | FastAPI backend |
| Databases | SQLite SQL tool |
| Performance | Latency measurement per request/tool |
| Testing | Pytest + evaluation suite |
| Monitoring | Structured tool/agent logs and metrics |

## Run locally

```bash
pip install -r requirements.txt
python main.py "What is Retrieval-Augmented Generation?"
python main.py "How many learners are in Bengaluru?"
python main.py "What is 47 * 6 + 12?"
```

Optional live LLM mode:

```powershell
$env:ANTHROPIC_API_KEY="your_key"
python main.py "Explain how Agentic AI differs from a fixed RAG pipeline."
```

Run tests:

```bash
pytest -q
python eval/evaluate.py
```

Run API:

```bash
uvicorn src.api:app --reload
```

Then POST JSON to `/api/query`:

```json
{"query":"What is RAG?"}
```

## Design principle

The offline path is deterministic and testable. The live path adds LLM-based routing and grounded response generation. External APIs are optional, so core orchestration and evaluation remain reproducible.
