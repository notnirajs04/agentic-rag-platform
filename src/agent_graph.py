"""LangGraph multi-agent orchestration for the Agentic RAG platform."""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from tools import TOOLS, retrieve_documents, query_database, analyze_data, calculator

MAX_STEPS = 8

class AgentState(TypedDict, total=False):
    question: str
    route: str
    steps: int
    tool_log: list[dict[str, Any]]
    evidence: list[str]
    draft_answer: str
    final_answer: str
    validation: dict[str, Any]
    done: bool
    started_at: float


def offline_route(question: str) -> str:
    q = question.lower()
    if any(x in q for x in ["average", "mean", "highest score", "top score", "analyze", "dataset"]):
        return "data"
    if any(x in q for x in ["select ", "database", "sql", "how many learners", "how many students"]):
        return "sql"
    if re.search(r"\d+\s*[+\-*/]\s*\d+", q):
        return "calculator"
    return "rag"


def supervisor_node(state: AgentState) -> AgentState:
    question = state["question"]
    use_live = bool(os.getenv("ANTHROPIC_API_KEY"))
    if not use_live:
        route = offline_route(question)
    else:
        route = llm_route(question)
    return {**state, "route": route, "steps": state.get("steps", 0) + 1}


def llm_route(question: str) -> str:
    """Use an LLM only for routing; tool execution remains controlled by the graph."""
    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        max_tokens=40,
        system=("Classify the user request into exactly one route: rag, sql, data, calculator. "
                "Return only the route name. Use rag for document knowledge questions, sql for database queries, "
                "data for statistical/data-analysis requests, and calculator for arithmetic."),
        messages=[{"role": "user", "content": question}],
    )
    text = "".join(getattr(b, "text", "") for b in response.content).strip().lower()
    return text if text in {"rag", "sql", "data", "calculator"} else "rag"


def rag_agent(state: AgentState) -> AgentState:
    started = time.perf_counter()
    result = retrieve_documents.invoke(state["question"])
    latency = (time.perf_counter() - started) * 1000
    return _record(state, "rag_agent", state["question"], result, latency)


def sql_agent(state: AgentState) -> AgentState:
    q = state["question"].lower()
    if "how many" in q and ("learner" in q or "student" in q):
        sql = "SELECT COUNT(*) AS learner_count FROM learners"
    elif "bengaluru" in q:
        sql = "SELECT COUNT(*) AS learner_count FROM learners WHERE city = 'Bengaluru'"
    elif "average" in q or "mean" in q:
        sql = "SELECT AVG(score) AS average_score FROM learners"
    else:
        sql = "SELECT * FROM learners LIMIT 10"
    started = time.perf_counter()
    result = query_database.invoke(sql)
    latency = (time.perf_counter() - started) * 1000
    return _record(state, "sql_agent", sql, result, latency)


def data_agent(state: AgentState) -> AgentState:
    started = time.perf_counter()
    result = analyze_data.invoke(state["question"])
    latency = (time.perf_counter() - started) * 1000
    return _record(state, "data_agent", state["question"], result, latency)


def calculator_agent(state: AgentState) -> AgentState:
    expr = re.search(r"[\d+\-*/().\s]{3,}", state["question"])
    expression = expr.group(0).strip() if expr else state["question"]
    started = time.perf_counter()
    result = calculator.invoke(expression)
    latency = (time.perf_counter() - started) * 1000
    return _record(state, "calculator_agent", expression, result, latency)


def _record(state: AgentState, agent: str, tool_input: str, result: str, latency_ms: float) -> AgentState:
    log = list(state.get("tool_log", []))
    log.append({"agent": agent, "input": tool_input, "result": result, "latency_ms": round(latency_ms, 3)})
    evidence = list(state.get("evidence", []))
    evidence.append(result)
    return {**state, "tool_log": log, "evidence": evidence, "steps": state.get("steps", 0) + 1}


def response_node(state: AgentState) -> AgentState:
    evidence = "\n\n".join(state.get("evidence", []))
    question = state["question"]
    if os.getenv("ANTHROPIC_API_KEY"):
        from anthropic import Anthropic
        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        prompt = ("Answer the user using only the supplied evidence. Be concise and explicit when evidence is missing.\n\n"
                  f"Question: {question}\nEvidence:\n{evidence}")
        response = client.messages.create(model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"), max_tokens=300,
                                          system="You are the response agent in a grounded multi-agent system.",
                                          messages=[{"role": "user", "content": prompt}])
        answer = "".join(getattr(b, "text", "") for b in response.content).strip()
    else:
        answer = evidence.split("\n\n")[-1] if evidence else "No evidence was produced."
    return {**state, "draft_answer": answer, "steps": state.get("steps", 0) + 1}


def validation_node(state: AgentState) -> AgentState:
    answer = state.get("draft_answer", "")
    evidence = " ".join(state.get("evidence", []))
    grounded = bool(answer) and (any(token.lower() in evidence.lower() for token in answer.split() if len(token) > 5) or not evidence)
    validation = {"grounded": grounded, "evidence_count": len(state.get("evidence", [])), "passed": bool(answer) and grounded}
    return {**state, "validation": validation, "final_answer": answer if validation["passed"] else "Validation failed: insufficient grounded evidence.", "done": True, "steps": state.get("steps", 0) + 1}


def route_after_supervisor(state: AgentState) -> Literal["rag", "sql", "data", "calculator", "response"]:
    if state.get("steps", 0) >= MAX_STEPS:
        return "response"
    return state.get("route", "rag")


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("rag", rag_agent)
    graph.add_node("sql", sql_agent)
    graph.add_node("data", data_agent)
    graph.add_node("calculator", calculator_agent)
    graph.add_node("response", response_node)
    graph.add_node("validation", validation_node)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_after_supervisor,
                                {"rag": "rag", "sql": "sql", "data": "data", "calculator": "calculator", "response": "response"})
    for node in ["rag", "sql", "data", "calculator"]:
        graph.add_edge(node, "response")
    graph.add_edge("response", "validation")
    graph.add_edge("validation", END)
    return graph.compile()


def run_agent(question: str) -> AgentState:
    initial: AgentState = {"question": question, "steps": 0, "tool_log": [], "evidence": [], "draft_answer": "", "final_answer": "", "validation": {}, "done": False, "started_at": time.time()}
    return build_graph().invoke(initial)
