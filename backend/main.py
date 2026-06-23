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


def rebuild_db():
    global db

    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    db = build_db()
