"""Standalone terminal chat against the same vector store the API uses."""
import ollama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

from config import LLM_MODEL

load_dotenv()

per_dir = "db/chromadb"


def generate_answer(query, docs, chat_history):
    context = "\n\n".join(
        f"[Source {i+1}]: {doc.page_content}" for i, doc in enumerate(docs)
    )
    prompt = f"""Answer using ONLY the context below.
    If the answer isn't there, say "I don't have enough information."
    Context:
    {context}"""

    messages = [{"role": "system", "content": prompt}]
    messages.extend(chat_history[-20:])
    messages.append({"role": "user", "content": query})

    response = ollama.chat(model=LLM_MODEL, messages=messages)
    return response.message.content


def main():
    chat_history = []

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(
        persist_directory=per_dir,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    base_retriever = db.as_retriever(search_kwargs={"k": 8})

    while True:
        try:
            query = input("Ask your Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            continue
        if query.lower() == "exit":
            break

        # retrieve the docs
        candidate_docs = base_retriever.invoke(query)

        # Reranking the candidate docs
        top_docs = []
        if candidate_docs:
            pairs = [[query, doc.page_content] for doc in candidate_docs]
            scores = reranker.predict(pairs)
            ranked = sorted(
                zip(scores, candidate_docs), key=lambda x: x[0], reverse=True
            )
            top_docs = [doc for _, doc in ranked[:3]]

        answer = generate_answer(query, top_docs, chat_history)

        # Save to history
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer})

        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()
