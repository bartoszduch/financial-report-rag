import json
from pathlib import Path
from typing import Iterable

from src.models import DocumentChunk


def recall_at_k(retrieved_ids: Iterable[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    retrieved = list(retrieved_ids)[:k]
    if not relevant_ids:
        return 0.0
    return len(set(retrieved).intersection(relevant_ids)) / len(relevant_ids)


def evaluate_retrieval(dataset: dict, retrieval_fn) -> dict:
    per_query = []
    for item in dataset.get("queries", []):
        relevant = set(item.get("relevant_chunk_ids", []))
        retrieved_chunks: list[DocumentChunk] = retrieval_fn(item["question"], 5)
        ids = [chunk.chunk_id for chunk in retrieved_chunks]
        per_query.append({
            "id": item["id"],
            "question": item["question"],
            "retrieved_chunk_ids": ids,
            "relevant_chunk_ids": sorted(relevant),
            "recall_at_1": recall_at_k(ids, relevant, 1),
            "recall_at_3": recall_at_k(ids, relevant, 3),
            "recall_at_5": recall_at_k(ids, relevant, 5),
        })

    count = len(per_query)
    if not count:
        return {
            "retrieval": {"recall_at_1": 0.0, "recall_at_3": 0.0, "recall_at_5": 0.0,
                          "evaluated_queries": 0, "per_query": []}
        }
    return {
        "retrieval": {
            "recall_at_1": sum(x["recall_at_1"] for x in per_query) / count,
            "recall_at_3": sum(x["recall_at_3"] for x in per_query) / count,
            "recall_at_5": sum(x["recall_at_5"] for x in per_query) / count,
            "evaluated_queries": count,
            "per_query": per_query,
        }
    }


def evaluate_answers(dataset: dict, answer_fn, embedder=None) -> dict:
    unsupported_expected = "Insufficient information in the provided document."
    rows = []
    for item in dataset.get("queries", []):
        answer = answer_fn(item["question"])
        unsupported = item.get("unsupported", False)
        refusal_correct = unsupported and answer == unsupported_expected
        citations_valid = bool(answer == unsupported_expected or __import__("re").search(r"\[Source \d+\]", answer))
        exact_numbers_valid = all(
            str(number) in answer
            for number in item.get("expected_numbers", [])
        )
        currency_consistent = all(
            currency in answer
            for currency in item.get("expected_currencies", [])
        )
        semantic_similarity = None
        if embedder is not None and item.get("reference_answer"):
            vectors = embedder.encode_documents([answer, item["reference_answer"]])
            semantic_similarity = float(vectors[0] @ vectors[1])
        rows.append({
            "id": item["id"],
            "unsupported": unsupported,
            "refusal_correct": refusal_correct,
            "citations_valid": citations_valid,
            "exact_numbers_valid": exact_numbers_valid,
            "currency_consistent": currency_consistent,
            "semantic_similarity": semantic_similarity,
        })
    return {"answer": {"evaluated_queries": len(rows), "per_query": rows}}


def run_evaluation(dataset_path: str | Path, retrieval_fn, answer_fn=None, embedder=None) -> dict:
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    report = evaluate_retrieval(dataset, retrieval_fn)
    if answer_fn:
        report.update(evaluate_answers(dataset, answer_fn, embedder))
    return report
