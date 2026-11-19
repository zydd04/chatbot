"""Settings shared by the backend modules."""
import os

DOCS_PATH = "docs"
DB_PATH = "db/chromadb"
EVAL_SET_PATH = "eval/eval_set.json"
# fixed documents the eval set was written against, kept apart from docs/ so
# uploading or deleting files in the app can't break the eval
EVAL_CORPUS_PATH = "eval/corpus"
REPORT_PATH = "eval/latest_report.json"

LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:0.5b")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

SUPPORTED_EXTENSIONS = (".txt", ".pdf", ".docx")
COLLECTION_METADATA = {"hnsw:space": "cosine"}

# Generation runs on CPU, where reading the prompt is the slow part, so the
# prompt is kept small: short chunks, few of them, and a short chat history.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75
TOP_N_CHUNKS = 3
MAX_HISTORY_MESSAGES = 6
# 0 = always pick the most likely next token: answers stick closer to the
# context, and the same question gives the same answer (repeatable evals)
LLM_TEMPERATURE = 0
# keep the model in memory between questions instead of reloading it
LLM_KEEP_ALIVE = "30m"
