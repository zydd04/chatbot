"""Talking to the local LLM through Ollama."""
import threading
import time

import ollama

from config import LLM_MODEL, LLM_KEEP_ALIVE, LLM_TEMPERATURE

# set once the model has answered (or been loaded), so the UI can say why
# the first reply is slow
ready = threading.Event()
# time-to-first-token of the first generation since startup
cold_start_seconds = None


def prepare(retries: int = 60, delay: float = 5.0):
    """
    Downloads the model if it is missing and loads it into memory, so the
    first chat message doesn't have to wait for either. Retries because the
    ollama container may still be starting when the backend comes up.
    """
    for _ in range(retries):
        try:
            try:
                ollama.show(LLM_MODEL)
            except ollama.ResponseError as e:
                if e.status_code != 404:
                    raise
                print(f"Downloading {LLM_MODEL}...")
                ollama.pull(LLM_MODEL)

            # an empty prompt just loads the model
            ollama.generate(model=LLM_MODEL, prompt="", keep_alive=LLM_KEEP_ALIVE)
            ready.set()
            print(f"{LLM_MODEL} is ready")
            return
        except Exception as e:
            print(f"LLM not ready yet: {e}")
            time.sleep(delay)


def build_system_prompt(docs) -> str:
    context = "\n\n".join(
        f"[Source {i+1}]\n{d.page_content}"
        for i, d in enumerate(docs)
    )

    return (
        "You are a strict retrieval assistant.\n"
        "Answer ONLY using the provided context.\n"
        "If the answer is not explicitly contained in the context, "
        "reply exactly:\n"
        "I don't have enough information.\n"
        "Do not use prior knowledge.\n\n"
        f"Context:\n{context}"
    )


def _chat(messages):
    return ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        stream=True,
        keep_alive=LLM_KEEP_ALIVE,
        options={"temperature": LLM_TEMPERATURE},
    )


def _stream(messages):
    try:
        yield from _chat(messages)
    except ollama.ResponseError as e:
        # model not pulled yet (fresh ollama container): pull once and retry.
        # A 404 is raised before any chunk is produced, so retrying is safe.
        if e.status_code != 404:
            raise
        ollama.pull(LLM_MODEL)
        yield from _chat(messages)


def generate(messages):
    """Yields answer text pieces; records time-to-first-token of the first call."""
    global cold_start_seconds
    start = time.time()

    for chunk in _stream(messages):
        if cold_start_seconds is None:
            cold_start_seconds = time.time() - start
        ready.set()
        text = chunk["message"]["content"]
        if text:
            yield text
