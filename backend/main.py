from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import json
import time
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain import EnsembleRetriever
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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
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
all_chuncks_cache =[]
_process_start = time.time()
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
    if path.endswith(".txt"):
        return TextLoader(path, encoding="utf-8").load()

    if path.endswith(".pdf"):
        return PyPDFLoader(path).load()

    if path.endswith(".docx"):
        text = docx2txt.process(path)
        return [type("Doc", (), {"page_content": text, "metadata": {"source": path}})]

    return []


def load_all_docs():
    docs = []

    if not os.path.exists(DOCS_PATH):
        os.makedirs(DOCS_PATH)

    for file in os.listdir(DOCS_PATH):
        path = os.path.join(DOCS_PATH, file)
        docs.extend(load_file(path))

    return docs

##chunking ...

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    return splitter.split_documents(docs)

def build_indexes():
    """
    Hybrid Search: Chroma for dense embedding search, 
    BM25 for keyword search
    both synched
    """
    global all_chunks_cache
 
    docs = load_all_docs()
    chunks = split_docs(docs)
    all_chunks_cache = chunks
 
    if not chunks:
        dense = Chroma(
            embedding_function=embedding_model,
            persist_directory=DB_PATH,
            collection_metadata={"hnsw:space": "cosine"},
        )
        sparse = None
        return dense, sparse
 
    dense = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=DB_PATH,
        collection_metadata={"hnsw:space": "cosine"}
    )
 
    sparse = BM25Retriever.from_documents(chunks)
    sparse.k = 8
 
    return dense, sparse
def load_db():
    dense = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_model
    )
    docs = load_all_docs()
    chunks = split_docs(docs)
    global all_chunks_cache
    all_chunks_cache = chunks
 
    sparse = None
    if chunks:
        sparse = BM25Retriever.from_documents(chunks)
        sparse.k = 8
 
    return dense, sparse

def add_file_to_db(path):
    #switch to mb25 + cache
    global bm25_retriever, all_chunks_cache
 
    docs = load_file(path)
    chunks = split_docs(docs)
 
    db.add_documents(chunks)
 
    all_chunks_cache = all_chunks_cache + chunks
    bm25_retriever = BM25Retriever.from_documents(all_chunks_cache)
    bm25_retriever.k = 8


##STARTUP...

@app.on_event("startup")
def startup():
    global db, bm25_retriever
 
    if os.path.exists(DB_PATH):
        db, bm25_retriever = load_db()
    else:
        db, bm25_retriever = build_indexes()
 
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)


##Retrieve + Rerank
def retrieve(query: str):
    

##Chat...
@app.post("/chat")
async def chat(req: ChatRequest):

    docs = retrieve(req.message)

    context = "\n\n".join(
        f"[Source {i+1}]\n{d.page_content}"
        for i, d in enumerate(docs)
    )

    system = f"""
    You are a strict retrieval assistant.

    Answer ONLY using the provided context.

    If the answer is not explicitly contained in the context,
    reply exactly:

    I don't have enough information.

    Do not use prior knowledge.

    Context:
    {context}
    """

    messages = [{"role": "system", "content": system}]

    for m in req.history[-20:]:
        messages.append({"role": m.role, "content": m.content})

    messages.append({"role": "user", "content": req.message})

    def stream():
        response = ollama.chat(
            model="llama3.2:1b",
            messages=messages,
            stream=True
        )

        for chunk in response:
            text = chunk["message"]["content"]

            yield json.dumps({
                "type": "chunk",
                "text": text
            }) + "\n"

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
async def upload(files: List[UploadFile] = File(...)):
    os.makedirs(DOCS_PATH, exist_ok=True)

    uploaded = []
    failed = []

    for file in files:
        try:
            path = os.path.join(DOCS_PATH, file.filename)

            with open(path, "wb") as f:
                f.write(await file.read())

            add_file_to_db(path)
            uploaded.append(file.filename)
        except Exception as e:
            failed.append({"file": file.filename, "error": str(e)})

    return {"uploaded": uploaded, "failed": failed}

##Listing Files..
@app.get("/files")
def list_files():
    os.makedirs(DOCS_PATH, exist_ok=True)
    return {"files": os.listdir(DOCS_PATH)}

##Delete...
@app.delete("/files/{filename}")
def delete_file(filename: str):
    global db, bm25_retriever
 
    path = os.path.join(DOCS_PATH, filename)
 
    if os.path.exists(path):
        os.remove(path)
 
    db, bm25_retriever = build_indexes()
 
    return {"message": "deleted"}
