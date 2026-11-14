from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import json
import time
import threading
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.retrievers import BM25Retriever
import docx2txt

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from sentence_transformers import CrossEncoder
import ollama

DOCS_PATH = "docs"
DB_PATH = "db/chromadb"
EVAL_SET_PATH = "eval/eval_set.json"
REPORT_PATH = "eval/latest_report.json"
LLM_MODEL = "qwen2.5:0.5b"
SUPPORTED_EXTENSIONS = (".txt", ".pdf", ".docx")
COLLECTION_METADATA = {"hnsw:space": "cosine"}
REFUSAL = "i don't have enough information"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

db = None
bm25_retriever = None
all_chunks_cache = []
index_lock = threading.Lock()
_cold_start_seconds = None

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []

class EvalCase(BaseModel):
    question : str
    expected_source : str
    unanswerable_bool : bool
    expected_keywords: List[str] = []

##File Loading ...
def load_file(path: str):
    lower = path.lower()

    if lower.endswith(".txt"):
        return TextLoader(path, encoding="utf-8").load()

    if lower.endswith(".pdf"):
        return PyPDFLoader(path).load()

    if lower.endswith(".docx"):
        text = docx2txt.process(path)
        return [Document(page_content=text, metadata={"source": path})]

    return []


def load_all_docs():
    docs = []

    os.makedirs(DOCS_PATH, exist_ok=True)

    for file in sorted(os.listdir(DOCS_PATH)):
        path = os.path.join(DOCS_PATH, file)
        if not os.path.isfile(path):
            continue
        try:
            docs.extend(load_file(path))
        except Exception as e:
            # one unreadable file must not take the whole server down
            print(f"Skipping {path}: {e}")

    return docs

##chunking ...

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    return splitter.split_documents(docs)


def open_dense():
    return Chroma(
        embedding_function=embedding_model,
        persist_directory=DB_PATH,
        collection_metadata=COLLECTION_METADATA,
    )


def build_bm25(chunks):
    if not chunks:
        return None
    sparse = BM25Retriever.from_documents(chunks)
    sparse.k = 8
    return sparse


def build_indexes(force: bool = False):
    """
    Hybrid Search: Chroma for dense embedding search,
    BM25 for keyword search
    both synched

    The persisted Chroma collection is reused when it already matches the
    files in docs/, otherwise it is wiped and rebuilt (re-adding on top of an
    existing collection would duplicate every chunk).
    """
    global all_chunks_cache

    chunks = split_docs(load_all_docs())
    all_chunks_cache = chunks

    dense = open_dense()

    if force or dense._collection.count() != len(chunks):
        dense.delete_collection()
        dense = open_dense()
        if chunks:
            dense.add_documents(chunks)

    return dense, build_bm25(chunks)


def remove_source_from_db(path):
    """Drop every chunk that came from `path` from both indexes."""
    global bm25_retriever, all_chunks_cache

    ids = db.get(where={"source": path})["ids"]
    if ids:
        db.delete(ids=ids)

    all_chunks_cache = [
        c for c in all_chunks_cache if c.metadata.get("source") != path
    ]
    bm25_retriever = build_bm25(all_chunks_cache)


def add_file_to_db(path):
    global bm25_retriever, all_chunks_cache

    chunks = split_docs(load_file(path))
    if not chunks:
        raise ValueError("No readable text found in file")

    # re-uploading a file replaces its old chunks instead of duplicating them
    remove_source_from_db(path)

    db.add_documents(chunks)

    all_chunks_cache = all_chunks_cache + chunks
    bm25_retriever = build_bm25(all_chunks_cache)


##STARTUP...

@app.on_event("startup")
def startup():
    global db, bm25_retriever

    db, bm25_retriever = build_indexes()

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)


##Hybrid Retrieve + Rerank
def reciprocal_rank_fusion(ranked_lists, k: int = 60):
    scores = {}
    doc_lookup = {}
 
    for ranked_docs in ranked_lists:
        for rank, doc in enumerate(ranked_docs):
            key = doc.page_content
            doc_lookup[key] = doc
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
 
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_lookup[key] for key, _ in merged]
 
 
def retrieve(query: str, k_dense: int = 8, k_sparse: int = 8, top_n: int = 3):
    dense_retriever = db.as_retriever(search_kwargs={"k": k_dense})
    dense_docs = dense_retriever.invoke(query)
 
    if bm25_retriever is not None:
        sparse_docs = bm25_retriever.invoke(query)
        docs = reciprocal_rank_fusion([dense_docs, sparse_docs])
    else:
        docs = dense_docs
 
    if not docs:
        return []
 
    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)
 
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
 
    return [d for _, d in ranked[:top_n]]
 
 
##Chat...
def build_system_prompt(docs) -> str:
    context = "\n\n".join(
        f"[Source {i+1}]\n{d.page_content}"
        for i, d in enumerate(docs)
    )

    return f"""
    You are a strict retrieval assistant.

    Answer ONLY using the provided context.

    If the answer is not explicitly contained in the context,
    reply exactly:

    I don't have enough information.

    Do not use prior knowledge.

    Context:
    {context}
    """


def _ollama_stream(messages):
    try:
        yield from ollama.chat(model=LLM_MODEL, messages=messages, stream=True)
    except ollama.ResponseError as e:
        # model not pulled yet (fresh ollama container): pull once and retry.
        # A 404 is raised before any chunk is produced, so retrying is safe.
        if e.status_code != 404:
            raise
        ollama.pull(LLM_MODEL)
        yield from ollama.chat(model=LLM_MODEL, messages=messages, stream=True)


def generate(messages):
    """Yields answer text pieces; records time-to-first-token of the first call."""
    global _cold_start_seconds
    start = time.time()

    for chunk in _ollama_stream(messages):
        if _cold_start_seconds is None:
            _cold_start_seconds = time.time() - start
        text = chunk["message"]["content"]
        if text:
            yield text


@app.post("/chat")
def chat(req: ChatRequest):
    docs = retrieve(req.message)

    messages = [{"role": "system", "content": build_system_prompt(docs)}]

    for m in req.history[-20:]:
        if m.role in ("user", "assistant") and m.content:
            messages.append({"role": m.role, "content": m.content})

    messages.append({"role": "user", "content": req.message})

    def stream():
        try:
            for text in generate(messages):
                yield json.dumps({
                    "type": "chunk",
                    "text": text
                }) + "\n"
        except Exception as e:
            yield json.dumps({
                "type": "error",
                "error": f"LLM request failed: {e}"
            }) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
            return

        yield json.dumps({
            "type": "sources",
            "sources": [
                {
                    "file": d.metadata.get("source", "unknown"),
                    "preview": d.page_content[:200]
                }
                for d in docs
            ]
        }) + "\n"

        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")

##Uploads... (now accepts multiple files in one request)
@app.post("/upload")
def upload(files: List[UploadFile] = File(...)):
    os.makedirs(DOCS_PATH, exist_ok=True)

    uploaded = []
    failed = []

    for file in files:
        # strip any directory part so a crafted name can't escape docs/
        name = os.path.basename((file.filename or "").replace("\\", "/"))

        if not name or not name.lower().endswith(SUPPORTED_EXTENSIONS):
            failed.append({
                "file": name or "unknown",
                "error": "Unsupported file type (use .txt, .pdf or .docx)",
            })
            continue

        path = os.path.join(DOCS_PATH, name)

        try:
            with open(path, "wb") as f:
                f.write(file.file.read())

            with index_lock:
                add_file_to_db(path)
            uploaded.append(name)
        except Exception as e:
            # don't leave a file listed that isn't searchable
            with index_lock:
                try:
                    remove_source_from_db(path)
                except Exception:
                    pass
            if os.path.exists(path):
                os.remove(path)
            failed.append({"file": name, "error": str(e)})

    return {"uploaded": uploaded, "failed": failed}

#chat synchronization
def chat_sync(message: str) -> dict:
    """
    used by the eval harness to measure latency per request and inspect the full
    response + sources in one shot.
    """
    docs = retrieve(message)

    answer = "".join(generate([
        {"role": "system", "content": build_system_prompt(docs)},
        {"role": "user", "content": message},
    ]))
    sources = [d.metadata.get("source", "unknown") for d in docs]

    return {"answer": answer, "sources": sources}

#report generating
def run_eval() -> dict:
    if not os.path.exists(EVAL_SET_PATH):
        return {"error": f"No eval set found at {EVAL_SET_PATH}"}
 
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        raw_cases = json.load(f)
    cases = [EvalCase(**c) for c in raw_cases]
 
    if not cases:
        return {"error": "Eval set is empty."}
 
    retrieval_hits = 0
    answerable = 0
    hallucinations = 0
    latencies = []
    case_results = []
 
    for case in cases:
        start = time.time()
        result = chat_sync(case.question)
        elapsed = time.time() - start
        latencies.append(elapsed)
 
        # unanswerable cases have no source to find, so they don't count
        # towards recall
        if case.unanswerable_bool:
            retrieved_hit = None
        else:
            answerable += 1
            retrieved_hit = any(case.expected_source in s for s in result["sources"])
            if retrieved_hit:
                retrieval_hits += 1
 
        answer_lower = result["answer"].lower().replace("\u2019", "'")
        refused = REFUSAL in answer_lower
 
        if case.unanswerable_bool and not refused:
            is_hallucination = True
        else:
            if not case.expected_keywords:
                is_hallucination = False
            else:
                is_hallucination = not any(
                    kw.lower() in answer_lower for kw in case.expected_keywords
                )
 
        if is_hallucination:
            hallucinations += 1
 
        case_results.append({
            "question": case.question,
            "answer": result["answer"],
            "sources": result["sources"],
            "retrieval_hit": retrieved_hit,
            "flagged_hallucination": is_hallucination,
            "latency_seconds": round(elapsed, 3),
        })
 
    latencies.sort()
    n = len(latencies)
    p50 = latencies[int(n * 0.5)] if n else None
    p95 = latencies[min(int(n * 0.95), n - 1)] if n else None
 
    report = {
        "num_cases": n,
        "recall_at_10": round(retrieval_hits / answerable, 3) if answerable else None,
        "hallucination_rate": round(hallucinations / n, 3) if n else None,
        "latency_p50_ms": round(p50 * 1000, 1) if p50 is not None else None,
        "latency_p95_ms": round(p95 * 1000, 1) if p95 is not None else None,
        "cold_start_seconds": round(_cold_start_seconds, 2) if _cold_start_seconds is not None else None,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cases": case_results,
    }
 
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
 
    return report
 
#run reporting...
@app.post("/report/run")
def report_run():
    return run_eval()

#get latest report
@app.get("/report")
def report_latest():
    """Returns the most recently generated report without re-running eval."""
    if not os.path.exists(REPORT_PATH):
        return {"error": "No report generated yet. POST /report/run first."}
    with open(REPORT_PATH, encoding="utf-8") as f:
        return json.load(f)

##Listing Files..
@app.get("/files")
def list_files():
    os.makedirs(DOCS_PATH, exist_ok=True)
    return {
        "files": sorted(
            f for f in os.listdir(DOCS_PATH)
            if os.path.isfile(os.path.join(DOCS_PATH, f))
        )
    }

##Delete...
@app.delete("/files/{filename}")
def delete_file(filename: str):
    name = os.path.basename(filename)
    path = os.path.join(DOCS_PATH, name)

    if not name or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")

    with index_lock:
        os.remove(path)
        remove_source_from_db(path)

    return {"message": "deleted"}
