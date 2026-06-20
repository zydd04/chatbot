import ollama
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

query = input("Ask your Question: ")

#retrieve the docs
base_retriever = db.as_retriever(search_kwargs={"k": 8})
candidate_docs = base_retriever.invoke(query)

#Reranking the candidate docs
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
pairs = [[query, doc.page_content] for doc in candidate_docs]
scores = reranker.predict(pairs)

ranked = sorted(zip(scores, candidate_docs), key=lambda x: x[0], reverse=True)
top_docs = [doc for _, doc in ranked[:3]]

#LLM implementatiom
def generate_answer(query, docs):
    context = "\n\n".join([f"[Source {i+1}]: {doc.page_content}" for i, doc in enumerate(docs)])
    prompt = f"""Answer using ONLY the context below.
    If the answer isn't there, say "I don't have enough information."
    Context:
    {context}
    Question: {query}
    Answer:"""
    response = ollama.chat(
        model="llama3.2:1b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.message.content

answer = generate_answer(query, top_docs)
print(f"Response: {answer}")