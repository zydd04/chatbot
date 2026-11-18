"""Loading files from disk and splitting them into chunks."""
import os

import docx2txt
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DOCS_PATH, CHUNK_SIZE, CHUNK_OVERLAP


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


def load_all_docs(folder: str = DOCS_PATH):
    docs = []

    os.makedirs(folder, exist_ok=True)

    for file in sorted(os.listdir(folder)):
        path = os.path.join(folder, file)
        if not os.path.isfile(path):
            continue
        try:
            docs.extend(load_file(path))
        except Exception as e:
            # one unreadable file must not take the whole server down
            print(f"Skipping {path}: {e}")

    return docs


def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)
