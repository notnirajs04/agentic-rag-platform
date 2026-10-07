import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from agent_graph import run_agent

def test_rag_route():
    r=run_agent("What is Retrieval-Augmented Generation?")
    assert r["route"] == "rag"
    assert r["validation"]["passed"]

def test_sql_route():
    r=run_agent("How many learners are in the database?")
    assert r["route"] == "sql"
    assert "6" in r["final_answer"]

def test_data_route():
    r=run_agent("What is the average score?")
    assert r["route"] == "data"
    assert "87" in r["final_answer"]

def test_calculator_route():
    r=run_agent("What is 47 * 6 + 12?")
    assert r["route"] == "calculator"
    assert "294" in r["final_answer"]
