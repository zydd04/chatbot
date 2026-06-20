from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv()

per_dir = "db/chromadb"
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(
    persist_directory=per_dir,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

query = "Who is the Founder of Google?"

#Fetch more candidates
base_retriever = db.as_retriever(search_kwargs={"k": 8})
candidate_docs = base_retriever.invoke(query)

#Rerank using CrossEncoder directly
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
pairs = [[query, doc.page_content] for doc in candidate_docs]
scores = reranker.predict(pairs)

#Sort by score and take top 3
ranked = sorted(zip(scores, candidate_docs), key=lambda x: x[0], reverse=True)
top_docs = [doc for _, doc in ranked[:3]]

print(f"Query: {query}")
print("Context:")
for i, (score, doc) in enumerate(ranked[:3], 1):
    print(f"Answer {i}:\n{doc.page_content}\n")