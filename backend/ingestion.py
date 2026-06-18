import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter 
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

#Load Files
def load_docs(doc_path="docs"):
    print(f"Loading documents from the folder {doc_path} ...")

    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"The Directory {doc_path} does not exist.")
    
    load = DirectoryLoader(path=doc_path, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    docs = load.load()
    
    if len(docs) == 0:
        raise FileNotFoundError(f"No .txt File found in the Folder {doc_path}")

    for i, doc in enumerate(docs[:2]):
        print(f"\nDocument {i+1}:")
        print(f"    Source:{doc.metadata['source']}")
    
    return docs
def main():
    docs = load_docs(doc_path="docs")

if __name__ == "__main__" :
    main()
