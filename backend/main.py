from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import os
import json
import shutil

from langchain_community.document_loaders import TextLoader, PyPDFLoader
import docx2txt

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from sentence_transformers import CrossEncoder
import ollama

DOCS_PATH = "docs"
DB_PATH = "db/chromadb"

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

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []

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

##Vector DB...

def build_db():
    docs = load_all_docs()
    chunks = split_docs(docs)

    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=DB_PATH,
        collection_metadata={"hnsw:space": "cosine"}
    )


def load_db():
    return Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_model
    )


def add_file_to_db(path):
    docs = load_file(path)
    chunks = split_docs(docs)

    db.add_documents(chunks)

##STARTUP...

@app.on_event("startup")
def startup():
    global db

    if os.path.exists(DB_PATH):
        db = load_db()
    else:
        db = build_db()

##Retrieve + Rerank
def retrieve(query: str):
    retriever = db.as_retriever(search_kwargs={"k": 8})
    docs = retriever.invoke(query)

    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    return [d for _, d in ranked[:3]]

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

##Uploads...
@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    os.makedirs(DOCS_PATH, exist_ok=True)

    path = os.path.join(DOCS_PATH, file.filename)

    with open(path, "wb") as f:
        f.write(await file.read())

    rebuild_db()

    return {"message": "uploaded"}

##Listing Files..
@app.get("/docs")
def list_files():
    os.makedirs(DOCS_PATH, exist_ok=True)
    return {"files": os.listdir(DOCS_PATH)}

##Delete...
@app.delete("/docs/{filename}")
def delete_file(filename: str):

    path = os.path.join(DOCS_PATH, filename)

    if os.path.exists(path):
        os.remove(path)

    rebuild_db()

    return {"message": "deleted"}