
from __future__ import annotations

import time
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel, Field

from agent_graph import run_agent

app = FastAPI(
    title="Agentic RAG & Multi-Agent Q&A Platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/query")
def query(request: QueryRequest):
    started = time.perf_counter()

    result = run_agent(request.query)

    latency_ms = (time.perf_counter() - started) * 1000

    return {
        "answer": result.get("final_answer", ""),
        "route": result.get("route"),
        "validation": result.get("validation", {}),
        "steps": result.get("steps", []),
        "latency_ms": round(latency_ms, 2),
        "agents": [
            call["agent"]
            for call in result.get("tool_calls", [])
            if "agent" in call
        ],
    }
