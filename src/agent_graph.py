r"""
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
- Optional OpenAI LLM planning via LangChain ChatOpenAI
- LangGraph stateful orchestration
- RAG retrieval
- SQL database queries
- Pandas data analysis
- Calculator tool
- Multi-step workflows
- Grounded response generation
- Response validation
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI


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

    supervisor_decision: Literal["continue", "finish"]
    next_agent: AgentName
    next_task: str
    supervisor_reason: str
    supervisor_history: list[dict[str, Any]]

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
        plan = [
            {
                "agent": "sql",
                "task": "Find the scores of learners located in Bengaluru.",
            },
            {
                "agent": "data",
                "task": "Calculate the average score of the Bengaluru learners.",
            },
        ]

        if any(
            phrase in q
            for phrase in [
                "explain",
                "available documents",
                "supporting documents",
                "what does the result tell us",
            ]
        ):
            plan.append(
                {
                    "agent": "rag",
                    "task": (
                        "Retrieve relevant documents that help explain "
                        "the Bengaluru learner results. Use only relevant "
                        "retrieved evidence."
                    ),
                }
            )

        return plan
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
    # Default â†’ RAG
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
    Use OpenAI via LangChain as an optional planner.

    The project remains fully functional without an API k   ey.
    """

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        return None

    try:
        llm = ChatOpenAI(
            model=os.getenv(
                "OPENROUTER_MODEL",
                "nvidia/nemotron-3.5-super-vl:free",
            ),
            temperature=0,
            max_tokens=160,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
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


        start_time = time.perf_counter()
        response = llm.invoke(prompt)
        print(f"[TIMING] Planner LLM: {time.perf_counter() - start_time:.2f}s")

        text = response.content

        if not isinstance(text, str):
            text = str(text)

        print("[Planner DEBUG] Response type:", type(text).__name__)
        print("[Planner DEBUG] Response preview:", repr(text[:500]))

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

    except Exception as exc:
        print(
            f"[OpenRouter Planner] Failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return None


# ---------------------------------------------------------------------------
# Planner Node
# ---------------------------------------------------------------------------



def planner_node(state: AgentState) -> AgentState:
    """Create a plan using deterministic routing first."""

    question = state["question"]
    q = question.lower().strip()

    # Identify known multi-agent workflows.
    is_multistep = (
        "bengaluru" in q
        and ("average" in q or "mean" in q)
        and ("learner" in q or "student" in q)
    ) or (
        ("database" in q or "learners" in q or "students" in q)
        and (
            "explain" in q
            or "according to" in q
            or "what does this mean" in q
        )
    )

    if is_multistep:
        plan = offline_plan(question)
        print("[Planner] Using deterministic multi-agent plan")
    else:
        # offline_plan already handles RAG, SQL, data and calculator.
        plan = offline_plan(question)

        # Keep the LLM planner available as an optional fallback
        # only if deterministic planning produces no usable plan.
        if not plan:
            plan = llm_plan(question)

        print("[Planner] Using deterministic single-agent plan")

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
    - "How many learners are in Bengaluru?" â†’ COUNT = 3
    - "Find scores of learners in Bengaluru" â†’ return rows
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
    print(
    f"[Agent Execution] Step {current_step + 1}/{len(plan)}: "
    f"{agent}"
)
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
# Supervisor Agent
# ---------------------------------------------------------------------------

def supervisor_node(state: AgentState) -> AgentState:
    """
    Decide whether the workflow should continue with another specialist
    agent or proceed to response generation.

    The supervisor uses the existing execution plan and collected evidence
    to make a deterministic decision. The LLM-based planner remains
    responsible for creating the initial plan.
    """

    question = state.get("question", "")
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    evidence = state.get("evidence", [])

    completed_steps = current_step + 1

    # ---------------------------------------------------------------
    # No evidence â†’ finish and let response generation handle it.
    # ---------------------------------------------------------------

    if not evidence:
        return {
            "supervisor_decision": "finish",
            "supervisor_reason": (
                "No evidence was produced by the current execution step."
            ),
        }

    # ---------------------------------------------------------------
    # Continue with the next planned specialist.
    # ---------------------------------------------------------------

    next_step = current_step + 1

    if next_step < len(plan):
        next_plan = plan[next_step]

        return {
            "supervisor_decision": "continue",
            "next_agent": next_plan["agent"],
            "next_task": next_plan["task"],
            "supervisor_reason": (
                f"Step {completed_steps} produced evidence, "
                f"but the planned workflow requires another "
                f"specialist step."
            ),
        }

    # ---------------------------------------------------------------
    # All planned work is complete.
    # ---------------------------------------------------------------

    return {
        "supervisor_decision": "finish",
        "supervisor_reason": (
            f"Completed {completed_steps} specialist step(s). "
            "The planned workflow has collected the required evidence."
        ),
    }

def llm_supervisor(state: AgentState) -> AgentState:
    """
    Use OpenAI to decide whether the workflow should continue
    with another specialist agent or finish.

    The supervisor is constrained by the existing execution plan.
    It may continue to the next planned specialist, but it must
    finish once all planned steps have been completed.
    """

    current_step = state.get(
        "current_step",
        0,
    )

    plan = state.get(
        "plan",
        [],
    )

    evidence = state.get(
        "evidence",
        [],
    )

    question = state.get(
        "question",
        "",
    )

    # ------------------------------------------------------------------
    # Deterministic safety rule:
    # If all planned steps have already been executed, finish.
    # ------------------------------------------------------------------

    next_step = current_step + 1

    if next_step >= len(plan):
        return {
            "supervisor_decision": "finish",
            "supervisor_reason": (
                "All planned specialist steps have been completed."
            ),
        }

    # ------------------------------------------------------------------
    # If an existing plan contains another specialist step,
    # prefer that planned step rather than allowing the LLM
    # to repeatedly select the same agent.
    # ------------------------------------------------------------------

    next_plan = plan[next_step]

    next_agent = next_plan.get(
        "agent"
    )

    next_task = next_plan.get(
        "task"
    )

    if not next_agent or not next_task:
        return {
            "supervisor_decision": "finish",
            "supervisor_reason": (
                "No valid next specialist step is available."
            ),
        }

    # ------------------------------------------------------------------
    # Optional LLM supervision.
    # The LLM can decide whether the next planned step is actually
    # necessary, but it cannot invent an unbounded sequence.
    # ------------------------------------------------------------------

    api_key = os.getenv(
        "OPENROUTER_API_KEY"
    )

    if not api_key:
        return {
            "supervisor_decision": "continue",
            "next_agent": next_agent,
            "next_task": next_task,
            "supervisor_reason": (
                "The next planned specialist step is required."
            ),
        }

    try:
        llm = ChatOpenAI(
            model=os.getenv(
                "OPENROUTER_MODEL",
                "nvidia/nemotron-3.5-super-vl:free",
            ),
            temperature=0,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        evidence_text = "\n\n".join(
            evidence
        )

        prompt = f"""
You are the supervisor of a multi-agent AI system.

User question:
{question}

Current completed step:
{current_step + 1}

Remaining planned specialist step:
Agent: {next_agent}
Task: {next_task}

Evidence collected so far:
{evidence_text}

Decide whether the remaining planned step is necessary.

IMPORTANT RULES:
1. You may only continue with the specialist agent listed above.
2. Do not invent another agent.
3. Do not repeat an agent that has already completed its planned task.
4. If the remaining planned step is needed, return "continue".
5. If the evidence already answers the user's question, return "finish".
6. Once there are no remaining planned steps, the workflow must finish.

Return exactly this JSON:

{{
  "decision": "continue" or "finish",
  "reason": "short explanation"
}}
"""

        response = llm.invoke(
            prompt
        )

        output = response.content

        if not isinstance(
            output,
            str,
        ):
            output = str(output)

        match = re.search(
            r"\{.*\}",
            output,
            flags=re.DOTALL,
        )

        if not match:
            return {
                "supervisor_decision": "continue",
                "next_agent": next_agent,
                "next_task": next_task,
                "supervisor_reason": (
                    "Proceeding with the next planned specialist step."
                ),
            }

        payload = json.loads(
            match.group(0)
        )

        decision = str(
            payload.get(
                "decision",
                "continue",
            )
        ).lower().strip()

        reason = _clean_text(
            payload.get(
                "reason",
                "Supervisor evaluated the next planned step.",
            )
        )

        if decision == "finish":
            return {
                "supervisor_decision": "finish",
                "supervisor_reason": reason,
            }

        return {
            "supervisor_decision": "continue",
            "next_agent": next_agent,
            "next_task": next_task,
            "supervisor_reason": reason,
        }

    except Exception:
        return {
            "supervisor_decision": "continue",
            "next_agent": next_agent,
            "next_task": next_task,
            "supervisor_reason": (
                "Proceeding with the next planned specialist step."
            ),
        }

# ---------------------------------------------------------------------------
# Step Controller
# ---------------------------------------------------------------------------

def next_plan_step(state: AgentState) -> AgentState:
    """
    Move execution to the next specialist selected by the supervisor.

    If the supervisor-selected step already exists in the plan,
    reuse it instead of appending a duplicate step.
    """

    current_step = state.get(
        "current_step",
        0,
    )

    next_agent = state.get(
        "next_agent"
    )

    next_task = state.get(
        "next_task"
    )

    plan = list(
        state.get(
            "plan",
            [],
        )
    )

    # Prevent unbounded workflow loops.
    if current_step + 1 >= MAX_STEPS:
        return {
            "current_step": current_step + 1,
            "supervisor_decision": "finish",
            "supervisor_reason": (
                f"Maximum workflow limit of "
                f"{MAX_STEPS} steps reached."
            ),
        }

    if not next_agent or not next_task:
        return {
            "current_step": current_step + 1,
            "supervisor_decision": "finish",
            "supervisor_reason": (
                "No valid next specialist was selected."
            ),
        }

    next_step = current_step + 1

    # ---------------------------------------------------------------
    # If the next step already exists in the original plan,
    # do NOT append another copy.
    # ---------------------------------------------------------------
    if next_step < len(plan):
        existing_step = plan[next_step]

        return {
            "current_step": next_step,
            "route": existing_step["agent"],
        }

    # ---------------------------------------------------------------
    # Only append when the supervisor genuinely creates a new step.
    # ---------------------------------------------------------------
    plan.append({
        "agent": next_agent,
        "task": next_task,
    })

    return {
        "plan": plan,
        "current_step": next_step,
        "route": next_agent,
    }


def route_after_planner(
    state: AgentState,
) -> str:
    """Decide whether there is a plan to execute."""

    plan = state.get("plan", [])

    if not plan:
        return "response"

    return "execute_agent"


def route_after_supervisor(
    state: AgentState,
) -> str:
    """Route based on the supervisor's decision."""

    decision = state.get(
        "supervisor_decision",
        "finish",
    )

    if decision == "continue":
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
    # Bengaluru average + learner count
    # ---------------------------------------------------------------

    if "bengaluru" in question.lower() and any(
        word in question.lower()
        for word in ["average", "mean", "score"]
    ):
        learner_count = None
        average_score = None
        learner_details = []

        for item in evidence:
            # Extract learner records from SQL evidence
            if item.startswith("[SQL]"):
                match = re.search(
                    r"Result:\s*(\[[\s\S]*?\])",
                    item,
                )

                if match:
                    try:
                        import json

                        records = json.loads(match.group(1))

                        if isinstance(records, list):
                            learner_details = records
                            learner_count = len(records)
                    except (json.JSONDecodeError, TypeError):
                        pass

            # Extract average score from Data evidence
            if (
                item.startswith("[DATA]")
                and "Bengaluru average score" in item
            ):
                match = re.search(
                    r"Bengaluru average score:\s*([\d.]+)",
                    item,
                )

                if match:
                    average_score = match.group(1)

        if learner_count is not None and average_score is not None:
            answer = (
                f"There are {learner_count} learners in Bengaluru. "
                f"Their average score is {average_score}."
            )

            if learner_details:
                details = ", ".join(
                    f"{learner['name']} ({learner['score']:g})"
                    for learner in learner_details
                    if "name" in learner and "score" in learner
                )

                if details:
                    answer += f" The learners are: {details}."

            return answer

        if average_score is not None:
            return f"Bengaluru average score: {average_score}"


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
    """Use OpenAI via LangChain to synthesize the final answer."""

    api_key = os.getenv(
        "OPENROUTER_API_KEY"
    )

    if not api_key:
        return None

    try:
        llm = ChatOpenAI(
            model=os.getenv(
                "OPENROUTER_MODEL",
                "nvidia/nemotron-3.5-super-vl:free",
            ),
            temperature=0,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        context = "\n\n".join(
            evidence
        )
        print(f"[TIMING] Response context size: {len(context)} characters")

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


        start_time = time.perf_counter()
        response = llm.invoke(prompt)
        print(f"[TIMING] Response LLM: {time.perf_counter() - start_time:.2f}s")

        print(
            f"[OpenRouter] Model call succeeded: "
            f"{os.getenv('OPENROUTER_MODEL', 'nvidia/nemotron-3.5-super-vl:free')}"
        )

        output = response.content

        if not isinstance(output, str):
            output = str(output)

        return output.strip() or None

    except Exception as exc:
        print(f"[OpenRouter] Model call failed: {type(exc).__name__}: {exc}")
        return None



def response_node(
    state: AgentState,
) -> AgentState:
    """Prefer deterministic answers when sufficient structured evidence exists."""

    print("[DEBUG] response_node reached")
    print("[DEBUG] Question:", state.get("question"))
    print("[DEBUG] All evidence:", state.get("evidence", []))

    question = state["question"]
    evidence = state.get("evidence", [])

    # Detect whether the workflow produced retrieved document evidence.
    has_rag_evidence = any(
        isinstance(item, str) and item.startswith("[RAG]")
        for item in evidence
    )

    # Reuse the existing deterministic response logic.
    deterministic_answer = _deterministic_response(
        question,
        evidence,
    )

    # Use deterministic output only when no RAG synthesis is needed.
    if deterministic_answer and not has_rag_evidence:
        print("[Response] Using deterministic response")
        return {
            "answer": deterministic_answer,
            "final_answer": deterministic_answer,
        }

    # Preserve LLM synthesis for RAG and document-grounded workflows.
    answer = _llm_response(
        question,
        evidence,
    )

    if not answer:
        answer = deterministic_answer

    if not answer:
        answer = (
            "I couldn't generate a reliable answer from the available evidence."
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
    """Validate numerical consistency and evidence overlap."""

    answer = state.get("final_answer", "").strip()
    evidence = [
        item for item in state.get("evidence", [])
        if isinstance(item, str)
    ]
    tool_calls = state.get("tool_calls", [])

    passed = False
    reason = "No answer or supporting evidence available."
    checks = {}

    if answer and evidence:
        evidence_types = {
            item.split("]", 1)[0] + "]"
            for item in evidence
            if item.startswith("[")
        }

        has_calculator = "[CALCULATOR]" in evidence_types
        has_sql = "[SQL]" in evidence_types
        has_data = "[DATA]" in evidence_types
        has_rag = "[RAG]" in evidence_types

        # Normalize numeric values so 95 and 95.0 are equivalent.
        def extract_numbers(text):
            values = re.findall(
                r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?",
                text,
            )

            normalized = set()

            for value in values:
                try:
                    number = float(value.replace(",", ""))
                    normalized.add(format(number, ".10g"))
                except ValueError:
                    continue

            return normalized

        answer_numbers = extract_numbers(answer)
        evidence_numbers = extract_numbers("\n".join(evidence))

        # 1. Calculator: require its result in the final answer.
        if has_calculator:
            result_values = []

            for item in evidence:
                if item.startswith("[CALCULATOR]"):
                    match = re.search(r"Result:\s*(.+)", item)

                    if match:
                        result_values.append(match.group(1).strip())

            checks["calculator"] = bool(
                result_values
                and any(value in answer for value in result_values)
            )

        # 2. SQL/Data: check text overlap AND numerical consistency.
        if has_sql or has_data:
            structured_evidence = "\n".join(
                item for item in evidence
                if item.startswith("[SQL]") or item.startswith("[DATA]")
            )

            answer_tokens = {
                token.lower()
                for token in re.findall(r"\b\w+\b", answer)
                if len(token) > 3
            }

            structured_tokens = {
                token.lower()
                for token in re.findall(
                    r"\b\w+\b",
                    structured_evidence,
                )
                if len(token) > 3
            }

            checks["sql_data_overlap"] = bool(
                answer_tokens & structured_tokens
            )

            # Check numbers against all available evidence so that
            # document references or other supported numbers in RAG
            # answers are not automatically rejected.
            checks["numeric_consistency"] = (
                answer_numbers.issubset(evidence_numbers)
            )

        # 3. RAG: retain the existing lexical-overlap heuristic.
        if has_rag:
            rag_evidence = "\n".join(
                item for item in evidence
                if item.startswith("[RAG]")
            )

            rag_tokens = {
                token.lower()
                for token in re.findall(r"\b\w+\b", rag_evidence)
                if len(token) > 3
            }

            answer_tokens = {
                token.lower()
                for token in re.findall(r"\b\w+\b", answer)
                if len(token) > 3
            }

            overlap_ratio = (
                len(answer_tokens & rag_tokens) / len(answer_tokens)
                if answer_tokens else 0.0
            )

            checks["rag"] = overlap_ratio >= 0.15

        passed = bool(checks) and all(checks.values())

        failed_checks = [
            name for name, result in checks.items()
            if not result
        ]

        if passed:
            reason = (
                "All applicable checks passed. Numerical consistency "
                "and lexical overlap are heuristics, not proof of truth."
            )
        else:
            reason = (
                "Failed checks: " + ", ".join(failed_checks)
                if failed_checks
                else "No applicable checks could be performed."
            )

    validation = {
        "passed": passed,
        "grounded": passed,
        "reason": reason,
        "checks": checks,
        "evidence_count": len(evidence),
        "tool_call_count": len(tool_calls),
    }

    print(
        f"[Validation] passed={passed} | "
        f"checks={checks} | reason={reason}"
    )

    return {
        "validation": validation
    }

# ---------------------------------------------------------------------------
# Validation Routing and Correction
# ---------------------------------------------------------------------------

def route_after_validation(state: AgentState) -> str:
    """Route validated answers to END and failed answers to correction."""

    validation = state.get("validation", {})

    if validation.get("passed", False):
        print("[Validation Routing] Validation passed. Finishing workflow.")
        return "end"

    print("[Validation Routing] Validation failed. Routing to correction.")
    return "correction"


def correction_node(state: AgentState) -> AgentState:
    """Attempt a deterministic fallback when validation fails."""

    print("[Correction] Attempting deterministic fallback.")

    question = state.get("question", "")
    evidence = state.get("evidence", [])

    answer = _deterministic_response(
        question,
        evidence,
    )

    if answer:
        print("[Correction] Deterministic fallback produced an answer.")
    else:
        answer = (
            "I couldn't sufficiently verify the answer against the "
            "available evidence. Please refine the question or provide "
            "more relevant supporting information."
        )
        print("[Correction] No deterministic fallback was available.")

    return {
        "answer": answer,
        "final_answer": answer,
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
        "supervisor",
        supervisor_node,
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

    graph.add_node(
        "correction",
        correction_node,
    )

    # Start workflow.
    graph.add_edge(
        START,
        "planner",
    )

    # Planner selects the initial workflow.
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "execute_agent": "execute_agent",
            "response": "response",
        },
    )

    # Execute specialist agents.
    graph.add_edge(
        "execute_agent",
        "supervisor",
    )

    # Supervisor determines whether another agent is needed.
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "next_step": "next_step",
            "response": "response",
        },
    )

    # Continue to the next planned agent.
    graph.add_edge(
        "next_step",
        "execute_agent",
    )

    # Generate and validate the final answer.
    graph.add_edge(
        "response",
        "validation",
    )

    # Route based on the validation result.
    graph.add_conditional_edges(
        "validation",
        route_after_validation,
        {
            "end": END,
            "correction": "correction",
        },
    )

    # Finish after the fallback response.
    graph.add_edge(
        "correction",
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
        question
        route
        initial_route
        final_route
        plan
        final_answer
        evidence
        tool_calls
        validation
        steps
        supervisor_decisions
        error
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
        "supervisor_history": [],
    }

    result = GRAPH.invoke(
        initial_state
    )

    plan = result.get(
        "plan",
        [],
    )

    tool_calls = result.get(
        "tool_calls",
        [],
    )
    for call in tool_calls:
        print(
            f"[Agent Timing] {call.get('agent', 'unknown')}: "
            f"{call.get('latency_ms', 0)} ms"
        )

    final_route = result.get(
        "route",
        "unknown",
    )

    # Determine the first specialist agent selected by the planner.
    initial_route = (
        plan[0].get("agent", "unknown")
        if plan
        else "unknown"
    )

    # Build an execution trace from the plan and recorded tool calls.
    steps = []

    for index, step in enumerate(plan):
        steps.append({
            "step": index + 1,
            "agent": step.get("agent"),
            "task": step.get("task"),
        })

    supervisor_decisions = []

    supervisor_decision = result.get(
        "supervisor_decision"
    )

    supervisor_reason = result.get(
        "supervisor_reason"
    )

    if supervisor_decision:
        supervisor_decisions.append({
            "decision": supervisor_decision,
            "reason": supervisor_reason,
            "next_agent": result.get(
                "next_agent"
            ),
            "next_task": result.get(
                "next_task"
            ),
        })

    return {
        "question": question,

        # Final route is preserved for backward compatibility.
        "route": final_route,

        # New observability fields.
        "initial_route": initial_route,
        "final_route": final_route,

        "plan": plan,

        "final_answer": result.get(
            "final_answer",
            result.get("answer", ""),
        ),

        "evidence": result.get(
            "evidence",
            [],
        ),

        "tool_calls": tool_calls,

        "validation": result.get(
            "validation",
            {},
        ),

        "steps": steps,

        "supervisor_decisions": supervisor_decisions,

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
