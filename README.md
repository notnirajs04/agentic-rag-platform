# Agentic RAG & Multi-Agent Q&A Platform

An Agentic AI application built with **Python, LangChain, LangGraph, OpenRouter, RAG, specialized tools, FastAPI, React, and evaluation workflows**. The platform routes user questions to specialized agents for document retrieval, SQL queries, data analysis, and arithmetic, then validates the resulting answer.

## Architecture

**User → Planner/Supervisor → Specialized Agents → Response Generation → Validation → Final Answer**

Specialized agents:
- **RAG Agent:** Retrieves relevant context from a local document collection.
- **SQL Agent:** Executes read-only queries against a SQLite database.
- **Data Agent:** Performs structured data analysis using Pandas.
- **Calculator Agent:** Evaluates arithmetic expressions.
- **Response and Validation:** Produces answers from available evidence and checks grounding and numerical consistency.

## Core Capabilities

- **Agent orchestration:** Stateful workflows and conditional routing using LangGraph.
- **Tool integration:** Document retrieval, SQL, Pandas-based analysis, and arithmetic tools.
- **Retrieval-Augmented Generation:** TF-IDF-based document retrieval with top-k evidence selection.
- **LLM integration:** OpenRouter-compatible LLM access through LangChain's `ChatOpenAI` interface.
- **Structured data:** Read-only SQLite analytics through a dedicated SQL tool.
- **Data analysis:** Pandas-based calculations and analysis.
- **Validation:** Evidence-grounding and numerical-consistency checks.
- **FastAPI backend:** `/api/query` and `/health` endpoints.
- **React frontend:** Browser-based interface for submitting queries and viewing answers.
- **Evaluation:** Route accuracy, task success, validation status, execution steps, and latency.
- **Retrieval evaluation:** A separate evaluation script for retrieval hit rate and reciprocal rank.
- **Observability:** Agent execution logs and timing information.
- **Offline-first behavior:** Deterministic routing and response paths for supported queries.

## Technology Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Agent orchestration | LangGraph |
| LLM and tool integration | LangChain, OpenRouter |
| Retrieval | TF-IDF |
| Database | SQLite |
| Data analysis | Pandas |
| Backend | FastAPI, Uvicorn |
| Frontend | React, Vite |
| Testing | Pytest |
| Evaluation | Custom task and retrieval evaluation scripts |

## Project Structure

```text
agentic-rag-platform/
├── src/
│   ├── agent_graph.py
│   ├── tools.py
│   └── api.py
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.js
├── eval/
│   ├── evaluate.py
│   ├── evaluate_retrieval.py
│   └── test_set.json
├── tests/
├── outputs/
├── data/
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Prerequisites

- Python compatible with the project's dependencies
- Node.js and npm
- An OpenRouter API key for LLM-powered response generation

## Setup

### 1. Install Python dependencies

From the project root, activate your Python environment and run:

```powershell
pip install -r requirements.txt
```

### 2. Configure environment variables

Create your local `.env` file from the example:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set your own OpenRouter API key:

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=nvidia/nemotron-3.5-super-vl:free
```

Keep `.env` private. Do not commit API keys to GitHub. The `.env.example` file contains placeholders only.

### 3. Run the CLI

From the project root:

```powershell
python main.py "What is Retrieval-Augmented Generation?"
python main.py "How many learners are in Bengaluru?"
python main.py "What is 47 * 6 + 12?"
```

### 4. Start the backend API

Run these commands from the project root:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m uvicorn api:app --env-file .env --host 127.0.0.1 --port 8000
```

The API provides:

- `GET /health` — checks backend availability.
- `POST /api/query` — submits a question and returns the result.

Example request body:

```json
{
  "query": "What is RAG?"
}
```

### 5. Start the frontend

Open a second terminal and navigate to the frontend directory:

```powershell
Set-Location .\frontend
npm install
npm run dev
```

Open the local URL printed by Vite, typically `http://localhost:5173/`.

Keep the backend running while using the frontend.

## Testing

Run the automated test suite from the project root:

```powershell
python -m pytest -q
```

## Evaluation

Run the task evaluation suite:

```powershell
python eval/evaluate.py
```

Run the separate retrieval evaluation:

```powershell
python eval/evaluate_retrieval.py
```

The task evaluation measures routing and task completion against the configured test set. The retrieval evaluation measures document retrieval performance against its curated queries.

**Evaluation scope:** Retrieval metrics from a small, manually labelled test set should be interpreted as preliminary results, not as proof of general retrieval performance.

## Design Principles

- **Modular agents:** Separate responsibilities for retrieval, SQL, data analysis, and arithmetic.
- **Testability:** Automated tests and evaluation scripts support repeatable checks.
- **Grounding:** Answers are checked against the available evidence.
- **Observability:** Execution logs and latency measurements help identify slow stages.
- **Practical delivery:** A FastAPI backend exposes the workflow to a React frontend.

## Project Goal

This project demonstrates the design of an Agentic AI application that combines retrieval, specialized tools, structured data access, stateful orchestration, validation, evaluation, and API delivery.

It is intended to demonstrate skills relevant to **AI Engineering, Agentic AI, LLM applications, RAG systems, and backend AI development**.
