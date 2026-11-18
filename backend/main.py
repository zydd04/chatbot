"""FastAPI app: HTTP routes only. The work lives in the modules imported below."""
import json
import os
import threading
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import index
import llm
from config import DOCS_PATH, REPORT_PATH, SUPPORTED_EXTENSIONS, MAX_HISTORY_MESSAGES
from evaluation import run_eval

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []


@app.on_event("startup")
def startup():
    index.build_indexes()
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    # in the background so the API is reachable while the model downloads
    threading.Thread(target=llm.prepare, daemon=True).start()


def _line(**event) -> str:
    return json.dumps(event) + "\n"


@app.post("/chat")
def chat(req: ChatRequest):
    docs = index.retrieve(req.message)

    messages = [{"role": "system", "content": llm.build_system_prompt(docs)}]
    for m in req.history[-MAX_HISTORY_MESSAGES:]:
        if m.role in ("user", "assistant") and m.content:
            messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": req.message})

    def stream():
        # tell the UI why nothing is coming yet
        if not llm.ready.is_set():
            yield _line(type="status",
                        text="Preparing the model (first run can take a few minutes)")

        try:
            for text in llm.generate(messages):
                yield _line(type="chunk", text=text)
        except Exception as e:
            yield _line(type="error", error=f"LLM request failed: {e}")
            yield _line(type="done")
            return

        yield _line(type="sources", sources=[
            {"file": d.metadata.get("source", "unknown"), "preview": d.page_content[:200]}
            for d in docs
        ])
        yield _line(type="done")

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/upload")
def upload(files: List[UploadFile] = File(...)):
    os.makedirs(DOCS_PATH, exist_ok=True)

    uploaded = []
    failed = []

    for file in files:
        # strip any directory part so a crafted name can't escape docs/
        name = os.path.basename((file.filename or "").replace("\\", "/"))

        if not name or not name.lower().endswith(SUPPORTED_EXTENSIONS):
            failed.append({
                "file": name or "unknown",
                "error": "Unsupported file type (use .txt, .pdf or .docx)",
            })
            continue

        path = os.path.join(DOCS_PATH, name)

        try:
            with open(path, "wb") as f:
                f.write(file.file.read())

            with index.lock:
                index.add_file(path)
            uploaded.append(name)
        except Exception as e:
            # don't leave a file listed that isn't searchable
            with index.lock:
                try:
                    index.remove_source(path)
                except Exception:
                    pass
            if os.path.exists(path):
                os.remove(path)
            failed.append({"file": name, "error": str(e)})

    return {"uploaded": uploaded, "failed": failed}


@app.post("/report/run")
def report_run():
    return run_eval()


@app.get("/report")
def report_latest():
    """Returns the most recently generated report without re-running eval."""
    if not os.path.exists(REPORT_PATH):
        return {"error": "No report generated yet. POST /report/run first."}
    with open(REPORT_PATH, encoding="utf-8") as f:
        return json.load(f)


@app.get("/files")
def list_files():
    os.makedirs(DOCS_PATH, exist_ok=True)
    return {
        "files": sorted(
            f for f in os.listdir(DOCS_PATH)
            if os.path.isfile(os.path.join(DOCS_PATH, f))
        )
    }


@app.delete("/files/{filename}")
def delete_file(filename: str):
    name = os.path.basename(filename)
    path = os.path.join(DOCS_PATH, name)

    if not name or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")

    with index.lock:
        os.remove(path)
        index.remove_source(path)

    return {"message": "deleted"}
