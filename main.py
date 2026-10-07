import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from agent_graph import run_agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    args = parser.parse_args()
    result = run_agent(args.question)
    print("Route:", result.get("route"))
    print("Agents:", [x["agent"] for x in result.get("tool_log", [])])
    print("Answer:", result.get("final_answer"))
    print("Validation:", result.get("validation"))
    print("Steps:", result.get("steps"))
