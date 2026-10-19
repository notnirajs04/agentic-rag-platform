import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_graph import run_agent


ROOT = os.path.dirname(os.path.dirname(__file__))

with open(
    os.path.join(ROOT, "eval", "test_set.json"),
    encoding="utf-8",
) as f:
    cases = json.load(f)


results = []

route_ok = 0
task_ok = 0
multi_agent_count = 0


for i, case in enumerate(cases, start=1):
    r = run_agent(case["question"])

    answer = (
        r.get("final_answer") or ""
    ).lower()

    # The expected route represents the first specialist
    # selected by the planner.
    route = r.get(
        "initial_route",
        r.get("route"),
    )

    # Final route represents the last specialist agent
    # executed in the workflow.
    final_route = r.get(
        "final_route",
        r.get("route"),
    )

    steps = r.get(
        "steps",
        [],
    )

    agents = [
        step.get("agent")
        for step in steps
        if step.get("agent")
    ]

    is_multi_agent = len(agents) > 1

    if is_multi_agent:
        multi_agent_count += 1

    ok_route = route == case["expected_route"]

    ok_task = all(
        x.lower() in answer
        for x in case["expected_contains"]
    )

    route_ok += ok_route
    task_ok += ok_task

    results.append({
        "question": case["question"],
        "expected_route": case["expected_route"],
        "actual_route": route,
        "final_route": final_route,
        "route_correct": ok_route,
        "expected_contains": case["expected_contains"],
        "answer": r.get("final_answer"),
        "task_success": ok_task,
        "multi_agent": is_multi_agent,
        "agents": agents,
        "validation": r.get("validation"),
        "steps": steps,
        "tool_calls": r.get(
            "tool_calls",
            [],
        ),
        "supervisor_decisions": r.get(
            "supervisor_decisions",
            [],
        ),
        "error": r.get("error"),
    })

    status = "PASS" if ok_task else "FAIL"

    print(f"\n[{status}] Question {i}")
    print(f"Question : {case['question']}")
    print(f"Expected : {case['expected_route']}")
    print(f"Route    : {route}")
    print(f"Final    : {final_route}")
    print(f"Answer   : {r.get('final_answer')}")

    if agents:
        print(
            f"Agents   : {' -> '.join(agents)}"
        )

    if not ok_task:
        print(
            f"Expected text: "
            f"{case['expected_contains']}"
        )
        print(
            f"Validation   : "
            f"{r.get('validation')}"
        )
        print(
            f"Tool calls   : "
            f"{r.get('tool_calls', [])}"
        )


metrics = {
    "num_questions": len(cases),
    "route_accuracy": round(
        route_ok / len(cases),
        3,
    ),
    "task_success_rate": round(
        task_ok / len(cases),
        3,
    ),
    "multi_agent_questions": multi_agent_count,
}


os.makedirs(
    os.path.join(ROOT, "outputs"),
    exist_ok=True,
)


with open(
    os.path.join(
        ROOT,
        "outputs",
        "eval_results.json",
    ),
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        {
            "metrics": metrics,
            "results": results,
        },
        f,
        indent=2,
    )


print("\n" + "=" * 50)
print("EVALUATION SUMMARY")
print("=" * 50)
print(
    f"Questions       : "
    f"{metrics['num_questions']}"
)
print(
    f"Route accuracy  : "
    f"{metrics['route_accuracy']:.1%}"
)
print(
    f"Task success    : "
    f"{metrics['task_success_rate']:.1%}"
)
print(
    f"Multi-agent Qs  : "
    f"{metrics['multi_agent_questions']}"
)
print("=" * 50)