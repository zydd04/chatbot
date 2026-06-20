import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

#Function For Load Files
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
        print(f"Source:{doc.metadata['source']}")
        print(f"Content length: {len(doc.page_content)} characters.")
        print(f"Preview: {doc.page_content[:10]}")
    
    return docs

#Function for creating chunks
def split_docs(docs, chunk_size=1000, chunk_overlap=0):

    print("Generating chunks ...")
    text_split = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_split.split_documents(docs)

    if chunks:
    
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)
        
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")
    
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
    print(f"Vector store created with {vectorstore._collection.count()} documents.")
    return vectorstore

def main():
    # Config
    docs_path = "docs"
    persistent_directory = "db/chromadb"

    # Check if vector store already exists
    if os.path.exists(persistent_directory):
        print("Vector Store Exists. Loading...")
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"}
        )
        print(f"Loaded existing vector store with {vectorstore._collection.count()} documents.")
        return vectorstore

    # Load documents
    docs = load_docs(docs_path)
    # Split documents into chunks
    chunks = split_docs(docs)
    # Create vector store   
    vectorstore = create_vs(chunks, persistent_directory)
    return vectorstore

if __name__ == "__main__":
    main()