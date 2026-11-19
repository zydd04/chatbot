"""Eval harness: runs eval/eval_set.json against eval/corpus and scores it."""
import json
import os
import re
import time
import unicodedata
import uuid
from typing import List, Union

from langchain_chroma import Chroma
from pydantic import BaseModel

import index
import llm
from config import (
    EVAL_SET_PATH, EVAL_CORPUS_PATH, REPORT_PATH, COLLECTION_METADATA,
    TOP_N_CHUNKS,
)
from documents import load_all_docs, split_docs


class EvalCase(BaseModel):
    question: str
    expected_source: str
    unanswerable_bool: bool
    # every entry must appear in the answer; an entry that is a list is a set
    # of alternatives, any one of which is enough (e.g. ["eight", "8"])
    expected_keywords: List[Union[str, List[str]]] = []


# ways a small model says "it's not in the context" besides the exact phrase
# the prompt asks for; matched against normalized text (see _normalize)
REFUSAL_PATTERNS = [
    r"\b(do not|dont|does not|doesnt) have enough information\b",
    r"\bnot enough information\b",
    r"\b(is|are|was|were|isnt|arent)? ?not (explicitly )?"
    r"(stated|mentioned|provided|specified|given|included|contained|available)\b",
    r"\b(no|not any) (mention|information|data|details?)\b",
    r"\b(does not|doesnt|do not|dont) (say|state|mention|specify|provide|contain|include)\b",
    r"\b(cannot|can not|cant|unable to) (answer|determine|find|tell|say)\b",
]


def _normalize(text: str) -> str:
    """Lowercase, drop accents and punctuation, collapse spaces."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("’", "'").replace("'", "")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return f" {text.strip()} "


def _contains(normalized_answer: str, keyword: str) -> bool:
    # whole words only, so "26" doesn't match "260"
    return f" {_normalize(keyword).strip()} " in normalized_answer


def _matches(normalized_answer: str, expected) -> bool:
    alternatives = [expected] if isinstance(expected, str) else expected
    return any(_contains(normalized_answer, alt) for alt in alternatives)


def _is_refusal(normalized_answer: str) -> bool:
    return any(re.search(p, normalized_answer) for p in REFUSAL_PATTERNS)


def chat_sync(message: str, dense=None, sparse=None) -> dict:
    """Answers one question in one shot (no streaming), with its sources."""
    docs = index.retrieve(message, dense=dense, sparse=sparse)

    answer = "".join(llm.generate([
        {"role": "system", "content": llm.build_system_prompt(docs)},
        {"role": "user", "content": message},
    ]))
    sources = [d.metadata.get("source", "unknown") for d in docs]

    return {"answer": answer, "sources": sources}


def build_eval_indexes():
    """
    Indexes eval/corpus in a throwaway in-memory collection, so the eval
    always runs against the documents its questions were written for.
    """
    chunks = split_docs(load_all_docs(EVAL_CORPUS_PATH))
    if not chunks:
        raise ValueError(f"No documents found in {EVAL_CORPUS_PATH}")

    # no persist_directory -> in-memory; unique name so reruns don't collide
    dense = Chroma(
        collection_name=f"eval_{uuid.uuid4().hex}",
        embedding_function=index.embedding_model,
        collection_metadata=COLLECTION_METADATA,
    )
    dense.add_documents(chunks)
    return dense, index.build_bm25(chunks)


def _source_name(path: str) -> str:
    # sources may have been indexed with Windows separators
    return os.path.basename(path.replace("\\", "/"))


def score_case(case: EvalCase, result: dict) -> dict:
    answer = _normalize(result["answer"])

    if case.unanswerable_bool:
        # nothing to retrieve, so these don't count towards recall;
        # the only correct behaviour is refusing
        retrieval_hit = None
        refused = _is_refusal(answer)
        is_correct = refused
    else:
        retrieval_hit = any(
            _source_name(s) == case.expected_source for s in result["sources"]
        )
        # every expected keyword must appear (e.g. "when AND where"); an
        # answer that has them all is correct even if it also hedges
        is_correct = bool(case.expected_keywords) and all(
            _matches(answer, kw) for kw in case.expected_keywords
        )
        refused = not is_correct and _is_refusal(answer)

    return {
        "question": case.question,
        "answer": result["answer"],
        "sources": result["sources"],
        "retrieval_hit": retrieval_hit,
        "refused": refused,
        "correct": is_correct,
        # a refusal is a miss, not a hallucination: only an answer that
        # states something other than the expected facts counts
        "flagged_hallucination": not refused and not is_correct,
    }


def _percentile(sorted_values, q):
    n = len(sorted_values)
    return sorted_values[min(int(n * q), n - 1)] if n else None


def _ratio(part, whole):
    return round(part / whole, 3) if whole else None


def run_eval() -> dict:
    if not os.path.exists(EVAL_SET_PATH):
        return {"error": f"No eval set found at {EVAL_SET_PATH}"}

    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        cases = [EvalCase(**c) for c in json.load(f)]

    if not cases:
        return {"error": "Eval set is empty."}

    try:
        eval_dense, eval_sparse = build_eval_indexes()
    except Exception as e:
        return {"error": f"Could not index eval corpus: {e}"}

    results = []
    latencies = []
    try:
        for case in cases:
            start = time.time()
            try:
                result = chat_sync(case.question, eval_dense, eval_sparse)
            except Exception as e:
                # one failed request shouldn't throw away the whole run
                results.append({"question": case.question, "error": str(e)})
                continue
            elapsed = time.time() - start
            latencies.append(elapsed)

            scored = score_case(case, result)
            scored["answerable"] = not case.unanswerable_bool
            scored["latency_seconds"] = round(elapsed, 3)
            results.append(scored)
    finally:
        eval_dense.delete_collection()

    scored = [r for r in results if "error" not in r]
    answerable = [r for r in scored if r.pop("answerable")]
    latencies.sort()
    p50 = _percentile(latencies, 0.5)
    p95 = _percentile(latencies, 0.95)
    cold_start = llm.cold_start_seconds

    report = {
        "num_cases": len(cases),
        "num_errors": len(results) - len(scored),
        "k": TOP_N_CHUNKS,
        "recall_at_k": _ratio(sum(r["retrieval_hit"] for r in answerable), len(answerable)),
        "answer_accuracy": _ratio(sum(r["correct"] for r in answerable), len(answerable)),
        "hallucination_rate": _ratio(sum(r["flagged_hallucination"] for r in scored), len(scored)),
        "latency_p50_ms": round(p50 * 1000, 1) if p50 is not None else None,
        "latency_p95_ms": round(p95 * 1000, 1) if p95 is not None else None,
        "cold_start_seconds": round(cold_start, 2) if cold_start is not None else None,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cases": results,
    }

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report
