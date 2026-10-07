import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from agent_graph import run_agent

ROOT = os.path.dirname(os.path.dirname(__file__))
with open(os.path.join(ROOT, "eval", "test_set.json")) as f: cases = json.load(f)
results=[]; route_ok=0; task_ok=0
for case in cases:
    r=run_agent(case["question"]); answer=(r.get("final_answer") or "").lower()
    route = r.get("route"); ok_route = route == case["expected_route"]; ok_task=all(x.lower() in answer for x in case["expected_contains"])
    route_ok += ok_route; task_ok += ok_task
    results.append({"question":case["question"],"route":route,"route_correct":ok_route,"answer":r.get("final_answer"),"task_success":ok_task,"validation":r.get("validation"),"steps":r.get("steps")})
metrics={"num_questions":len(cases),"route_accuracy":round(route_ok/len(cases),3),"task_success_rate":round(task_ok/len(cases),3)}
os.makedirs(os.path.join(ROOT,"outputs"),exist_ok=True)
with open(os.path.join(ROOT,"outputs","eval_results.json"),"w") as f: json.dump({"metrics":metrics,"results":results},f,indent=2)
print(json.dumps(metrics,indent=2))
