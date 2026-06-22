from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import os
import subprocess
import json
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

#Function For Load Files
def load_docs(doc_path="docs"):
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"The Directory {doc_path} does not exist.")
    load = DirectoryLoader(path=doc_path, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    docs = load.load()
    if len(docs) == 0:
        raise FileNotFoundError(f"No .txt File found in the Folder {doc_path}")
    return docs

#Function for creating chunks
def split_docs(docs, chunk_size=1000, chunk_overlap=0):

    text_split = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_split.split_documents(docs)
    
    return chunks

# Function for creating vector store
def create_vs(chunks, persist_dir="db/chromadb"):
    print("Creating embeddings and storing in ChromaDB...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"}
    )
    return vectorstore

app = FastAPI()

#Connect api to frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WORKSPACE = "./workspace"
os.makedirs(WORKSPACE, exist_ok=True)

class Message(BaseModel):
    role: str   
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []

