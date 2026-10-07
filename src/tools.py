"""Tools used by the Agentic RAG platform.

The project intentionally keeps the default data local and deterministic so
that the complete workflow can be evaluated without an external API key.
"""
from __future__ import annotations

import ast
import operator as op
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import tool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "agent_data.db"

DOCUMENTS = [
    {"id": "doc-1", "title": "Agentic AI", "text": "Agentic AI systems can plan tasks, select tools, execute actions, observe results, and iterate toward a goal. LangGraph can represent these workflows as explicit stateful graphs."},
    {"id": "doc-2", "title": "Retrieval-Augmented Generation", "text": "Retrieval-Augmented Generation combines retrieval from a knowledge source with LLM generation. Retrieved context should be relevant to the query and should be used to ground the response."},
    {"id": "doc-3", "title": "RAG Evaluation", "text": "RAG systems should evaluate retrieval quality separately from answer quality. Useful retrieval metrics include hit rate, reciprocal rank, precision, and recall."},
    {"id": "doc-4", "title": "LangGraph", "text": "LangGraph supports stateful agent orchestration with nodes, conditional edges, loops, checkpoints, and human-in-the-loop patterns."},
    {"id": "doc-5", "title": "AI Reliability", "text": "Reliable AI applications should validate tool outputs, handle failures, record latency and usage metrics, and test routing and task success separately."},
]

_SAMPLE_ROWS = [
    (1, "Niraj", "Bengaluru", "AI", 92),
    (2, "Asha", "Mysuru", "Data", 86),
    (3, "Rahul", "Bengaluru", "AI", 88),
    (4, "Meera", "Chennai", "Software", 79),
    (5, "Arjun", "Bengaluru", "Data", 95),
    (6, "Diya", "Hyderabad", "Software", 83),
]

_vectorizer = TfidfVectorizer(stop_words="english")
_matrix = _vectorizer.fit_transform([d["text"] for d in DOCUMENTS])


def _safe_expression(expression: str) -> float:
    allowed = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
               ast.Pow: op.pow, ast.USub: op.neg, ast.UAdd: op.pos}
    def evaluate(node):
        if isinstance(node, ast.Expression): return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)): return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed: return allowed[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed: return allowed[type(node.op)](evaluate(node.operand))
        raise ValueError("Unsupported expression")
    return float(evaluate(ast.parse(expression, mode="eval")))


def initialize_database() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS learners (id INTEGER PRIMARY KEY, name TEXT, city TEXT, track TEXT, score REAL)")
        count = conn.execute("SELECT COUNT(*) FROM learners").fetchone()[0]
        if count == 0:
            conn.executemany("INSERT INTO learners VALUES (?, ?, ?, ?, ?)", _SAMPLE_ROWS)
        conn.commit()


@tool
def retrieve_documents(query: str) -> str:
    """Retrieve the most relevant knowledge-base passages for a question."""
    q = _vectorizer.transform([query])
    scores = cosine_similarity(q, _matrix)[0]
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:3]
    hits = []
    for idx, score in ranked:
        if score <= 0:
            continue
        doc = DOCUMENTS[idx]
        hits.append(f"[{doc['id']}] {doc['title']} (score={score:.3f}): {doc['text']}")
    return "\n".join(hits) if hits else "No relevant documents found."


@tool
def calculator(expression: str) -> str:
    """Safely evaluate a basic arithmetic expression."""
    try:
        return str(_safe_expression(expression))
    except Exception as exc:
        return f"Calculation error: {exc}"


@tool
def query_database(sql: str) -> str:
    """Execute a read-only SQL query against the local SQLite analytics database."""
    normalized = sql.strip().lower()
    if not normalized.startswith("select") or any(x in normalized for x in ["insert ", "update ", "delete ", "drop ", "alter ", "pragma "]):
        return "SQL rejected: only read-only SELECT queries are allowed."
    initialize_database()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(sql, conn)
        return df.to_json(orient="records")
    except Exception as exc:
        return f"SQL error: {exc}"


@tool
def analyze_data(operation: str) -> str:
    """Run a small Pandas-based analysis on the local learner dataset."""
    initialize_database()
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM learners", conn)
    op_name = operation.lower()
    if "average" in op_name or "mean" in op_name:
        return f"Average score: {df['score'].mean():.2f}"
    if "highest" in op_name or "top" in op_name:
        row = df.loc[df["score"].idxmax()]
        return f"Highest score: {row['name']} with {row['score']:.0f}"
    if "city" in op_name and "count" in op_name:
        return df["city"].value_counts().to_json()
    return f"Dataset has {len(df)} rows and columns: {', '.join(df.columns)}"


TOOLS = {
    "retrieve_documents": retrieve_documents,
    "calculator": calculator,
    "query_database": query_database,
    "analyze_data": analyze_data,
}

TOOL_DESCRIPTIONS = {name: tool.description for name, tool in TOOLS.items()}
