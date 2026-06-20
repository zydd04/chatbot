from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

per_dir = "db/chromadb"
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=per_dir, embedding_function=embedding_model, collection_metadata={"hnsw:space": "cosine"})

query = "When was Google First launched?"

retriever = db.as_retriever(search_kwargs={"k":3}) #retrieve the top 3 chunks 

similar_docs = retriever.invoke(query)

print(f"Query:  {query}")
print("Context: ")
for i, doc in enumerate(similar_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
