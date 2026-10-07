import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_graph import run_agent


ROOT = os.path.dirname(os.path.dirname(__file__))

with open(os.path.join(ROOT, "eval", "test_set.json")) as f:
    cases = json.load(f)


results = []
route_ok = 0
task_ok = 0

for i, case in enumerate(cases, start=1):
    r = run_agent(case["question"])

    answer = (r.get("final_answer") or "").lower()
    route = r.get("route")

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
        "route_correct": ok_route,
        "expected_contains": case["expected_contains"],
        "answer": r.get("final_answer"),
        "task_success": ok_task,
        "validation": r.get("validation"),
        "steps": r.get("steps"),
        "tool_log": r.get("tool_log", []),
    })

    status = "PASS" if ok_task else "FAIL"

    print(f"\n[{status}] Question {i}")
    print(f"Question : {case['question']}")
    print(f"Expected : {case['expected_route']}")
    print(f"Route    : {route}")
    print(f"Answer   : {r.get('final_answer')}")

    if not ok_task:
        print(f"Expected text: {case['expected_contains']}")
        print(f"Validation   : {r.get('validation')}")
        print(f"Tool log     : {r.get('tool_log', [])}")


metrics = {
    "num_questions": len(cases),
    "route_accuracy": round(route_ok / len(cases), 3),
    "task_success_rate": round(task_ok / len(cases), 3),
}


os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)

with open(
    os.path.join(ROOT, "outputs", "eval_results.json"),
    "w"
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
print(f"Questions       : {metrics['num_questions']}")
print(f"Route accuracy  : {metrics['route_accuracy']:.1%}")
print(f"Task success    : {metrics['task_success_rate']:.1%}")
print("=" * 50)