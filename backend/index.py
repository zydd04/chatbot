"""
Hybrid search over the uploaded documents: Chroma for dense embedding search,
BM25 for keyword search, merged with reciprocal rank fusion and reranked by a
cross-encoder.

The live indexes are module globals; read them as `index.db` etc. rather than
importing the names, so other modules see them after they are rebuilt.
"""
import threading

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from config import (
    DB_PATH, COLLECTION_METADATA, TOP_N_CHUNKS, EMBEDDING_MODEL, RERANKER_MODEL,
)
from documents import load_file, load_all_docs, split_docs

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
reranker = CrossEncoder(RERANKER_MODEL)

db = None
bm25_retriever = None
all_chunks_cache = []
# held while the indexes are being changed (uploads, deletes)
lock = threading.Lock()


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
    Loads both indexes from docs/. The persisted Chroma collection is reused
    when it already matches the files in docs/, otherwise it is wiped and
    rebuilt (re-adding on top of an existing collection would duplicate every
    chunk).
    """
    global db, bm25_retriever, all_chunks_cache

    chunks = split_docs(load_all_docs())
    all_chunks_cache = chunks

    dense = open_dense()

    if force or dense._collection.count() != len(chunks):
        dense.delete_collection()
        dense = open_dense()
        if chunks:
            dense.add_documents(chunks)

    db = dense
    bm25_retriever = build_bm25(chunks)


def remove_source(path):
    """Drop every chunk that came from `path` from both indexes."""
    global bm25_retriever, all_chunks_cache

    ids = db.get(where={"source": path})["ids"]
    if ids:
        db.delete(ids=ids)

    all_chunks_cache = [
        c for c in all_chunks_cache if c.metadata.get("source") != path
    ]
    bm25_retriever = build_bm25(all_chunks_cache)


def add_file(path):
    global bm25_retriever, all_chunks_cache

    chunks = split_docs(load_file(path))
    if not chunks:
        raise ValueError("No readable text found in file")

    # re-uploading a file replaces its old chunks instead of duplicating them
    remove_source(path)

    db.add_documents(chunks)

    all_chunks_cache = all_chunks_cache + chunks
    bm25_retriever = build_bm25(all_chunks_cache)


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


def retrieve(query: str, k_dense: int = 8, top_n: int = TOP_N_CHUNKS,
             dense=None, sparse=None):
    """Searches the app's indexes unless `dense`/`sparse` are given (eval)."""
    if dense is None:
        dense, sparse = db, bm25_retriever

    dense_docs = dense.as_retriever(search_kwargs={"k": k_dense}).invoke(query)

    if sparse is not None:
        docs = reciprocal_rank_fusion([dense_docs, sparse.invoke(query)])
    else:
        docs = dense_docs

    if not docs:
        return []

    scores = reranker.predict([[query, d.page_content] for d in docs])
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    return [d for _, d in ranked[:top_n]]
