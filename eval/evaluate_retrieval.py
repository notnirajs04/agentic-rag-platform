import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tools import retrieve_documents


# Each query has manually defined relevant document IDs.
# These labels are evaluation ground truth, not model predictions.
TEST_CASES = [
    {
        "query": "How do AI agents plan tasks and use tools?",
        "relevant_ids": ["doc-1"],
    },
    {
        "query": "What is Agentic AI and how does it work toward goals?",
        "relevant_ids": ["doc-1"],
    },
    {
        "query": "How does RAG combine retrieval with language model generation?",
        "relevant_ids": ["doc-2"],
    },
    {
        "query": "Why should retrieved context ground an answer?",
        "relevant_ids": ["doc-2"],
    },
    {
        "query": "Which metrics measure retrieval quality separately from answer quality?",
        "relevant_ids": ["doc-3"],
    },
    {
        "query": "What are hit rate and reciprocal rank used to evaluate?",
        "relevant_ids": ["doc-3"],
    },
    {
        "query": "What does LangGraph provide for stateful agent orchestration?",
        "relevant_ids": ["doc-4"],
    },
    {
        "query": "How can agent workflows use conditional edges, loops, and checkpoints?",
        "relevant_ids": ["doc-4"],
    },
    {
        "query": "How should reliable AI applications handle tool failures and validate outputs?",
        "relevant_ids": ["doc-5"],
    },
    {
        "query": "Why should AI systems track latency and test task success separately?",
        "relevant_ids": ["doc-5"],
    },
]


def retrieve_ids(query):
    result = retrieve_documents.invoke({"query": query})

    if not isinstance(result, str):
        result = str(result)

    return re.findall(r"\[(doc-\d+)\]", result)


def main():
    hits = 0
    reciprocal_ranks = []

    print("\nRETRIEVAL EVALUATION")
    print("=" * 72)

    for index, case in enumerate(TEST_CASES, start=1):
        retrieved_ids = retrieve_ids(case["query"])[:3]
        relevant_ids = set(case["relevant_ids"])

        first_relevant_rank = next(
            (
                rank
                for rank, doc_id in enumerate(retrieved_ids, start=1)
                if doc_id in relevant_ids
            ),
            None,
        )

        if first_relevant_rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / first_relevant_rank)
        else:
            reciprocal_ranks.append(0.0)

        status = "HIT" if first_relevant_rank is not None else "MISS"
        print(f"\n{index}. {case['query']}")
        print(f"   Relevant IDs : {sorted(relevant_ids)}")
        print(f"   Retrieved    : {retrieved_ids}")
        print(f"   First relevant rank: {first_relevant_rank}")
        print(f"   Result       : {status}")

    total = len(TEST_CASES)
    hit_rate = hits / total if total else 0.0
    mrr_at_3 = (
        sum(reciprocal_ranks) / total if total else 0.0
    )

    print("\n" + "=" * 72)
    print(f"Queries evaluated : {total}")
    print(f"Hit Rate@3        : {hit_rate:.3f} ({hit_rate:.1%})")
    print(f"MRR@3             : {mrr_at_3:.3f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
