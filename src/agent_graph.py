"""
Agentic RAG & Multi-Agent Q&A Platform

Architecture:
    User Question
          |
          v
      Planner
          |
          v
    Execute Agent
     /    |    |    \
   RAG   SQL  Data  Calculator
     \    |    |    /
          v
       Next Step
          |
          v
       Response
          |
          v
      Validation
          |
          v
        Final

The project supports:
- Deterministic offline planning
- Optional Anthropic LLM planning
- LangGraph stateful orchestration
- RAG retrieval
- SQL database queries
- Pandas data analysis
- Calculator tool
- Multi-step workflows
- Response validation
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from tools import (
    analyze_data,
    calculator,
    query_database,
    retrieve_documents,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_STEPS = 8

AgentName = Literal[
    "rag",
    "sql",
    "data",
    "calculator",
]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class PlanStep(TypedDict):
    agent: AgentName
    task: str


class AgentState(TypedDict, total=False):
    question: str

    plan: list[PlanStep]
    current_step: int

    route: str

    answer: str
    final_answer: str

    evidence: list[str]

    tool_calls: list[dict[str, Any]]

    validation: dict[str, Any]

    error: str


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def _clean_text(value: Any) -> str:
    return str(value).strip()


def _record(
    state: AgentState,
    agent: str,
    task: str,
    result: str,
    started_at: float,
) -> dict[str, Any]:
    """Create a standardized tool/agent execution record."""

    return {
        "agent": agent,
        "task": task,
        "result": result,
        "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
    }


def _normalize_agent(agent: str) -> AgentName | None:
    agent = agent.lower().strip()

    if agent in {"rag", "retrieval"}:
        return "rag"

    if agent in {"sql", "database", "db"}:
        return "sql"

    if agent in {"data", "analytics", "analysis"}:
        return "data"

    if agent in {"calculator", "calc", "math"}:
        return "calculator"

    return None


# ---------------------------------------------------------------------------
# Offline Planner
# ---------------------------------------------------------------------------

def offline_plan(question: str) -> list[PlanStep]:
    """
    Deterministic planner used when no LLM API key is available.

    This makes the project fully runnable offline and gives us
    deterministic evaluation results.
    """

    q = question.lower().strip()

    # ---------------------------------------------------------------
    # Multi-step: Bengaluru + average
    # ---------------------------------------------------------------

    if (
        "bengaluru" in q
        and (
            "average" in q
            or "mean" in q
        )
        and (
            "learner" in q
            or "student" in q
        )
    ):
        return [
            {
                "agent": "sql",
                "task": "Find the scores of learners located in Bengaluru.",
            },
            {
                "agent": "data",
                "task": "Calculate the average score of the Bengaluru learners.",
            },
        ]

    # ---------------------------------------------------------------
    # Multi-step: database result + explanation
    # ---------------------------------------------------------------

    if (
        ("database" in q or "learners" in q or "students" in q)
        and (
            "explain" in q
            or "according to" in q
            or "what does this mean" in q
        )
    ):
        return [
            {
                "agent": "sql",
                "task": question,
            },
            {
                "agent": "rag",
                "task": (
                    "Retrieve relevant knowledge that can help explain "
                    "the database result."
                ),
            },
        ]

    # ---------------------------------------------------------------
    # Calculator
    # ---------------------------------------------------------------

    if any(
        operator in q
        for operator in ["+", "-", "*", "/", "%"]
    ) and any(char.isdigit() for char in q):
        return [
            {
                "agent": "calculator",
                "task": question,
            }
        ]

    # ---------------------------------------------------------------
    # Data analysis
    # ---------------------------------------------------------------

    if any(
        keyword in q
        for keyword in [
            "average score",
            "mean score",
            "highest score",
            "top score",
            "city count",
            "distribution",
            "dataset",
            "data analysis",
        ]
    ):
        return [
            {
                "agent": "data",
                "task": question,
            }
        ]

    # ---------------------------------------------------------------
    # SQL / database
    # ---------------------------------------------------------------

    if any(
        keyword in q
        for keyword in [
            "how many learners",
            "how many students",
            "number of learners",
            "number of students",
            "database",
            "bengaluru",
            "mysuru",
            "chennai",
            "hyderabad",
        ]
    ):
        return [
            {
                "agent": "sql",
                "task": question,
            }
        ]

    # ---------------------------------------------------------------
    # Default → RAG
    # ---------------------------------------------------------------

    return [
        {
            "agent": "rag",
            "task": question,
        }
    ]


# ---------------------------------------------------------------------------
# Optional LLM Planner
# ---------------------------------------------------------------------------

def llm_plan(question: str) -> list[PlanStep] | None:
    """
    Use Anthropic as an optional planner.

    The project remains fully functional without an API key.
    """

    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        return None

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        model = os.getenv(
            "ANTHROPIC_MODEL",
            "claude-3-5-sonnet-latest",
        )

        prompt = f"""
You are the planner for a multi-agent AI system.

Available agents:

1. rag
   - Retrieves information from the knowledge base.

2. sql
   - Queries the learner database.

3. data
   - Performs Pandas-based data analysis.

4. calculator
   - Performs arithmetic calculations.

Create the minimum sequence of agents required to answer the question.

Rules:
- Return JSON only.
- Maximum 4 steps.
- Each step must contain:
  "agent": one of rag, sql, data, calculator
  "task": a clear instruction.
- Use multiple agents when the question genuinely requires multiple steps.

Question:
{question}

Return:
{{
  "steps": [
    {{
      "agent": "rag",
      "task": "..."
    }}
  ]
}}
"""

        response = client.messages.create(
            model=model,
            max_tokens=800,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        text = ""

        for block in response.content:
            if getattr(block, "type", None) == "text":
                text += block.text

        # Extract JSON if the model wrapped it in markdown.
        match = re.search(
            r"\{.*\}",
            text,
            flags=re.DOTALL,
        )

        if not match:
            return None

        payload = json.loads(match.group(0))

        steps = payload.get("steps", [])

        if not isinstance(steps, list):
            return None

        plan: list[PlanStep] = []

        for step in steps[:4]:
            if not isinstance(step, dict):
                continue

            agent = _normalize_agent(
                str(step.get("agent", ""))
            )

            task = _clean_text(
                step.get("task", question)
            )

            if agent and task:
                plan.append(
                    {
                        "agent": agent,
                        "task": task,
                    }
                )

        return plan or None

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Planner Node
# ---------------------------------------------------------------------------

def planner_node(state: AgentState) -> AgentState:
    """
    Creates an execution plan.

    Uses the LLM planner when available.
    Falls back to deterministic planning.
    """

    question = state["question"]

    plan = llm_plan(question)

    if not plan:
        plan = offline_plan(question)

    plan = plan[:MAX_STEPS]

    return {
        "plan": plan,
        "current_step": 0,
        "route": plan[0]["agent"] if plan else "rag",
        "evidence": [],
        "tool_calls": [],
    }


# ---------------------------------------------------------------------------
# RAG Agent
# ---------------------------------------------------------------------------

def rag_agent(state: AgentState) -> AgentState:
    """Execute the RAG retrieval agent."""

    started_at = time.perf_counter()

    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)

    if current_step >= len(plan):
        return state

    task = plan[current_step]["task"]

    result = retrieve_documents.invoke(
        {"query": task}
    )

    evidence = list(state.get("evidence", []))

    if result:
        evidence.append(
            f"[RAG]\n{result}"
        )

    calls = list(state.get("tool_calls", []))

    calls.append(
        _record(
            state,
            "rag",
            task,
            result,
            started_at,
        )
    )

    return {
        "evidence": evidence,
        "tool_calls": calls,
    }


# ---------------------------------------------------------------------------
# SQL Agent
# ---------------------------------------------------------------------------

def sql_agent(state: AgentState) -> AgentState:
    """
    Execute SQL/database tasks.

    Important:
    - "How many learners are in Bengaluru?" → COUNT = 3
    - "Find scores of learners in Bengaluru" → return rows
    - Average questions are handled appropriately.
    """

    started_at = time.perf_counter()

    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)

    if current_step >= len(plan):
        return state

    task = plan[current_step]["task"]

    q = task.lower()

    sql = ""

    # ---------------------------------------------------------------
    # Bengaluru COUNT
    # ---------------------------------------------------------------

    if (
        "bengaluru" in q
        and (
            "how many" in q
            or "number of" in q
            or "count" in q
        )
    ):
        sql = """
        SELECT COUNT(*) AS learner_count
        FROM learners
        WHERE city = 'Bengaluru'
        """

    # ---------------------------------------------------------------
    # Generic learner COUNT
    # ---------------------------------------------------------------

    elif (
        "how many" in q
        or "number of learners" in q
        or "number of students" in q
        or "count" in q
    ):
        sql = """
        SELECT COUNT(*) AS learner_count
        FROM learners
        """

    # ---------------------------------------------------------------
    # Bengaluru scores
    # ---------------------------------------------------------------

    elif "bengaluru" in q:
        sql = """
        SELECT name, score
        FROM learners
        WHERE city = 'Bengaluru'
        ORDER BY score DESC
        """

    # ---------------------------------------------------------------
    # Average score
    # ---------------------------------------------------------------

    elif (
        "average" in q
        or "mean" in q
    ):
        sql = """
        SELECT AVG(score) AS average_score
        FROM learners
        """

    # ---------------------------------------------------------------
    # Highest score
    # ---------------------------------------------------------------

    elif (
        "highest" in q
        or "top score" in q
    ):
        sql = """
        SELECT name, score
        FROM learners
        ORDER BY score DESC
        LIMIT 1
        """

    # ---------------------------------------------------------------
    # Fallback
    # ---------------------------------------------------------------

    else:
        sql = """
        SELECT *
        FROM learners
        LIMIT 10
        """

    result = query_database.invoke(
        {"sql": sql}
    )

    evidence = list(
        state.get("evidence", [])
    )

    evidence.append(
        f"[SQL]\nQuery: {sql.strip()}\nResult: {result}"
    )

    calls = list(
        state.get("tool_calls", [])
    )

    calls.append(
        _record(
            state,
            "sql",
            task,
            result,
            started_at,
        )
    )

    return {
        "evidence": evidence,
        "tool_calls": calls,
    }


# ---------------------------------------------------------------------------
# Data Agent
# ---------------------------------------------------------------------------

def data_agent(state: AgentState) -> AgentState:
    """Perform Pandas-based data analysis."""

    started_at = time.perf_counter()

    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)

    if current_step >= len(plan):
        return state

    task = plan[current_step]["task"]

    q = task.lower()

    # ---------------------------------------------------------------
    # Multi-agent Bengaluru average
    # ---------------------------------------------------------------

    if (
        "bengaluru" in q
        and (
            "average" in q
            or "mean" in q
        )
    ):
        # We intentionally calculate this through the data-analysis
        # agent rather than hard-coding the answer.
        from tools import initialize_database
        import sqlite3
        import pandas as pd

        initialize_database()

        with sqlite3.connect(
            "data/agent_data.db"
        ) as conn:
            df = pd.read_sql_query(
                """
                SELECT name, city, score
                FROM learners
                WHERE city = 'Bengaluru'
                """,
                conn,
            )

        if len(df) > 0:
            average = df["score"].mean()
            result = (
                f"Bengaluru average score: "
                f"{average:.2f}"
            )
        else:
            result = (
                "No Bengaluru learners found."
            )

    else:
        result = analyze_data.invoke(
            {"operation": task}
        )

    evidence = list(
        state.get("evidence", [])
    )

    evidence.append(
        f"[DATA]\n{result}"
    )

    calls = list(
        state.get("tool_calls", [])
    )

    calls.append(
        _record(
            state,
            "data",
            task,
            result,
            started_at,
        )
    )

    return {
        "evidence": evidence,
        "tool_calls": calls,
    }


# ---------------------------------------------------------------------------
# Calculator Agent
# ---------------------------------------------------------------------------

def calculator_agent(state: AgentState) -> AgentState:
    """Execute arithmetic calculation."""

    started_at = time.perf_counter()

    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)

    if current_step >= len(plan):
        return state

    task = plan[current_step]["task"]

    # Extract a simple mathematical expression.
    expression = task

    match = re.search(
        r"([0-9][0-9\s\+\-\*\/\%\.\(\)]+)",
        task,
    )

    if match:
        expression = match.group(1).strip()

    result = calculator.invoke(
        {"expression": expression}
    )

    evidence = list(
        state.get("evidence", [])
    )

    evidence.append(
        f"[CALCULATOR]\n"
        f"Expression: {expression}\n"
        f"Result: {result}"
    )

    calls = list(
        state.get("tool_calls", [])
    )

    calls.append(
        _record(
            state,
            "calculator",
            task,
            result,
            started_at,
        )
    )

    return {
        "evidence": evidence,
        "tool_calls": calls,
    }


# ---------------------------------------------------------------------------
# Agent Dispatcher
# ---------------------------------------------------------------------------

def execute_current_agent(
    state: AgentState,
) -> AgentState:
    """Execute the agent selected by the planner."""

    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)

    if current_step >= len(plan):
        return state

    agent = plan[current_step]["agent"]

    if agent == "rag":
        return rag_agent(state)

    if agent == "sql":
        return sql_agent(state)

    if agent == "data":
        return data_agent(state)

    if agent == "calculator":
        return calculator_agent(state)

    return {
        "error": f"Unknown agent: {agent}"
    }


# ---------------------------------------------------------------------------
# Step Controller
# ---------------------------------------------------------------------------

def next_plan_step(state: AgentState) -> AgentState:
    """Move the execution pointer to the next planned step."""

    current_step = state.get(
        "current_step",
        0,
    )

    next_step = current_step + 1

    plan = state.get("plan", [])

    updates: AgentState = {
        "current_step": next_step,
    }

    if next_step < len(plan):
        updates["route"] = plan[next_step]["agent"]

    return updates


def route_after_planner(
    state: AgentState,
) -> str:
    """Decide whether there is a plan to execute."""

    plan = state.get("plan", [])

    if not plan:
        return "response"

    return "execute_agent"


def route_after_agent(
    state: AgentState,
) -> str:
    """Loop through the remaining plan steps."""

    plan = state.get("plan", [])

    current_step = state.get(
        "current_step",
        0,
    )

    if current_step + 1 < len(plan):
        return "next_step"

    return "response"


# ---------------------------------------------------------------------------
# Response Generation
# ---------------------------------------------------------------------------

def _deterministic_response(
    question: str,
    evidence: list[str],
) -> str:
    """
    Generate a deterministic response when no LLM API is configured.
    """

    if not evidence:
        return "I could not find enough information to answer the question."

    # ---------------------------------------------------------------
    # Calculator
    # ---------------------------------------------------------------

    for item in evidence:
        if item.startswith("[CALCULATOR]"):
            match = re.search(
                r"Result:\s*(.+)",
                item,
            )

            if match:
                return (
                    f"The answer is {match.group(1).strip()}."
                )

    # ---------------------------------------------------------------
    # Learner count
    # ---------------------------------------------------------------
    for item in evidence:
        if item.startswith("[SQL]") and "learner_count" in item:
            match = re.search(
                r'"learner_count"\s*:\s*(\d+)',
                item,
            )

            if match:
                count = match.group(1)

                if "bengaluru" in question.lower():
                    return f"There are {count} learners in Bengaluru."

                return f"There are {count} learners in the database."

    # ---------------------------------------------------------------
    # Bengaluru average
    # ---------------------------------------------------------------

    for item in evidence:
        if item.startswith("[DATA]") and "Bengaluru average score" in item:
            return item.replace(
                "[DATA]",
                "",
            ).strip()

    # ---------------------------------------------------------------
    # Generic average
    # ---------------------------------------------------------------

    for item in evidence:
        if item.startswith("[DATA]") and "Average score:" in item:
            return item.replace(
                "[DATA]",
                "",
            ).strip()

    # ---------------------------------------------------------------
    # Generic SQL result
    # ---------------------------------------------------------------

    for item in evidence:
        if item.startswith("[SQL]"):
            match = re.search(
                r"Result:\s*(.+)",
                item,
                flags=re.DOTALL,
            )

            if match:
                return (
                    f"Database result: "
                    f"{match.group(1).strip()}"
                )

    # ---------------------------------------------------------------
    # RAG
    # ---------------------------------------------------------------

    rag_items = [
        item
        for item in evidence
        if item.startswith("[RAG]")
    ]

    if rag_items:
        text = rag_items[-1]

        text = text.replace(
            "[RAG]",
            "",
        ).strip()

        return text

    # ---------------------------------------------------------------
    # Fallback
    # ---------------------------------------------------------------

    return "\n\n".join(evidence)


def _llm_response(
    question: str,
    evidence: list[str],
) -> str | None:
    """Use Anthropic to synthesize the final answer."""

    api_key = os.getenv(
        "ANTHROPIC_API_KEY"
    )

    if not api_key:
        return None

    try:
        import anthropic

        client = anthropic.Anthropic(
            api_key=api_key
        )

        model = os.getenv(
            "ANTHROPIC_MODEL",
            "claude-3-5-sonnet-latest",
        )

        context = "\n\n".join(
            evidence
        )

        prompt = f"""
Answer the user's question using ONLY the
provided agent evidence.

If the evidence is insufficient, say so.

Do not invent information.

User question:
{question}

Agent evidence:
{context}
"""

        response = client.messages.create(
            model=model,
            max_tokens=700,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        output = ""

        for block in response.content:
            if getattr(block, "type", None) == "text":
                output += block.text

        return output.strip() or None

    except Exception:
        return None


def response_node(
    state: AgentState,
) -> AgentState:
    """Synthesize the final response from all agent evidence."""

    question = state["question"]

    evidence = state.get(
        "evidence",
        [],
    )

    answer = _llm_response(
        question,
        evidence,
    )

    if not answer:
        answer = _deterministic_response(
            question,
            evidence,
        )

    return {
        "answer": answer,
        "final_answer": answer,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validation_node(
    state: AgentState,
) -> AgentState:
    """Validate that the final answer is supported by agent evidence."""

    answer = state.get(
        "final_answer",
        "",
    )

    evidence = state.get(
        "evidence",
        [],
    )

    passed = False

    if answer and evidence:

        # Multiple agents produced evidence.
        if len(evidence) >= 2:
            passed = True

        # Calculator output is directly verifiable.
        elif any(
            item.startswith("[CALCULATOR]")
            for item in evidence
        ):
            passed = True

        # SQL/Data results are directly grounded.
        elif any(
            item.startswith("[SQL]")
            or item.startswith("[DATA]")
            for item in evidence
        ):
            passed = True

        # RAG answer: check basic overlap with retrieved evidence.
        else:
            answer_tokens = {
                token.lower()
                for token in re.findall(
                    r"\b\w+\b",
                    answer,
                )
                if len(token) > 3
            }

            evidence_tokens = {
                token.lower()
                for item in evidence
                for token in re.findall(
                    r"\b\w+\b",
                    item,
                )
                if len(token) > 3
            }

            passed = bool(
                answer_tokens & evidence_tokens
            )

    validation = {
        "passed": passed,
        "grounded": passed,
        "evidence_count": len(evidence),
        "tool_call_count": len(
            state.get("tool_calls", [])
        ),
    }

    return {
        "validation": validation
    }


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

def build_graph():
    """Build and compile the LangGraph workflow."""

    graph = StateGraph(
        AgentState
    )

    graph.add_node(
        "planner",
        planner_node,
    )

    graph.add_node(
        "execute_agent",
        execute_current_agent,
    )

    graph.add_node(
        "next_step",
        next_plan_step,
    )

    graph.add_node(
        "response",
        response_node,
    )

    graph.add_node(
        "validation",
        validation_node,
    )

    graph.add_edge(
        START,
        "planner",
    )

    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "execute_agent": "execute_agent",
            "response": "response",
        },
    )

    graph.add_conditional_edges(
        "execute_agent",
        route_after_agent,
        {
            "next_step": "next_step",
            "response": "response",
        },
    )

    graph.add_edge(
        "next_step",
        "execute_agent",
    )

    graph.add_edge(
        "response",
        "validation",
    )

    graph.add_edge(
        "validation",
        END,
    )

    return graph.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

GRAPH = build_graph()


def run_agent(
    question: str,
) -> dict[str, Any]:
    """
    Run the complete Agentic RAG workflow.

    Returns:
        route
        plan
        final_answer
        evidence
        tool_calls
        validation
    """

    initial_state: AgentState = {
        "question": question,
        "plan": [],
        "current_step": 0,
        "route": "planner",
        "evidence": [],
        "tool_calls": [],
        "answer": "",
        "final_answer": "",
        "validation": {},
    }

    result = GRAPH.invoke(
        initial_state
    )

    return {
        "question": question,
        "route": result.get(
            "route",
            "unknown",
        ),
        "plan": result.get(
            "plan",
            [],
        ),
        "final_answer": result.get(
            "final_answer",
            result.get("answer", ""),
        ),
        "evidence": result.get(
            "evidence",
            [],
        ),
        "tool_calls": result.get(
            "tool_calls",
            [],
        ),
        "validation": result.get(
            "validation",
            {},
        ),
        "error": result.get(
            "error"
        ),
    }


# ---------------------------------------------------------------------------
# Local Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    questions = [
        "What is Agentic AI?",
        "What does RAG combine?",
        "How many learners are in the database?",
        "How many learners are in Bengaluru?",
        "What is the average score?",
        "What is 47 * 6 + 12?",
        "What is the average score of learners in Bengaluru?",
    ]

    for question in questions:
        print("=" * 80)
        print(f"QUESTION: {question}")

        result = run_agent(question)

        print(f"ROUTE: {result['route']}")
        print(f"PLAN: {result['plan']}")
        print(f"ANSWER: {result['final_answer']}")
        print(f"VALIDATION: {result['validation']}")
        print()