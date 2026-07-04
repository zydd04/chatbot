// BUG 3 FIX: imports must come before all other statements in ES modules
import { useState, useEffect, useRef } from "react";
import "./App.css";

const API = "http://127.0.0.1:8000";

export default function App() {
  const [files, setFiles] = useState([]);
  const [message, setMessage] = useState("");
  const [history, setHistory] = useState([]);
  const bottomRef = useRef(null);
  const textareaRef = useRef(null);
  const [loading, setLoading] = useState(false);

  // Upload state: tracks in-flight upload + any per-file failures returned by the backend
  const [uploading, setUploading] = useState(false);
  const [uploadErrors, setUploadErrors] = useState([]);

  // Delete state: tracks which filename is currently being deleted + any error
  const [deletingFile, setDeletingFile] = useState(null);
  const [deleteError, setDeleteError] = useState(null);

  //File Handling
  const loadFiles = async () => {
    try {
      const res = await fetch(`${API}/docs`);
      const data = await res.json();
      setFiles(data.files || []);
    } catch (err) {
      console.error("File load error:", err);
    }
  };

  useEffect(() => {
    loadFiles();
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history]);

  //Upload handling (multiple files in a single request)
  const handleUpload = async (e) => {
    const selected = Array.from(e.target.files || []);
    if (selected.length === 0) return;

    const formData = new FormData();
    // Field name must match the backend param name: files: List[UploadFile]
    selected.forEach((file) => formData.append("files", file));

    setUploading(true);
    setUploadErrors([]);

    try {
      const res = await fetch(`${API}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Upload failed with status ${res.status}`);
      }

      const data = await res.json();

      if (data.failed && data.failed.length > 0) {
        setUploadErrors(data.failed);
      }

      await loadFiles();
    } catch (err) {
      console.error("Upload error:", err);
      setUploadErrors([{ file: "upload", error: err.message }]);
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  };

  //Delete handling
  const removeFile = async (fname) => {
    setDeletingFile(fname);
    setDeleteError(null);

    // Optimistically remove from UI so the button disappears immediately
    const previousFiles = files;
    setFiles((prev) => prev.filter((f) => f !== fname));

    try {
      const res = await fetch(`${API}/docs/${encodeURIComponent(fname)}`, {
        method: "DELETE",
      });

      if (!res.ok) {
        throw new Error(`Delete failed with status ${res.status}`);
      }

      await loadFiles();
    } catch (err) {
      console.error("Delete error:", err);
      setDeleteError(`Failed to delete "${fname}": ${err.message}`);
      // Roll back the optimistic removal since the delete didn't succeed
      setFiles(previousFiles);
    } finally {
      setDeletingFile(null);
    }
  };

  //Send handling
  const sendMessage = async () => {
    if (!message.trim() || loading) return;
    const userMsg = {
      role: "user",
      content: message.trim(),
    };
    const updatedHistory = [...history, userMsg];
    setHistory(updatedHistory);
    setMessage("");
    setLoading(true);
    setHistory((prev) => [
      ...prev,
      {
        role: "assistant",
        content: "",
        streaming: true,
        sources: [],
      },
    ]);
    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: userMsg.content,
          history: updatedHistory,
        }),
      });

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.trim()) continue;

          let data;
          try {
            data = JSON.parse(line);
          } catch {
            continue;
          }

          if (data.type === "chunk") {
            setHistory((prev) => {
              const updated = [...prev];
              const last = updated[updated.length - 1];
              if (last?.role === "assistant") {
                updated[updated.length - 1] = {
                  ...last,
                  content: last.content + data.text,
                };
              }
              return updated;
            });
          } else if (data.type === "sources") {
            setHistory((prev) => {
              const updated = [...prev];
              const last = updated[updated.length - 1];
              if (last?.role === "assistant") {
                updated[updated.length - 1] = {
                  ...last,
                  sources: data.sources,
                };
              }
              return updated;
            });
          } else if (data.type === "done") {
            setHistory((prev) => {
              const updated = [...prev];
              const last = updated[updated.length - 1];
              if (last?.role === "assistant") {
                updated[updated.length - 1] = {
                  ...last,
                  streaming: false,
                };
              }
              return updated;
            });
          }
        }
      }
    } catch (error) {
      setHistory((prev) => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last?.role === "assistant") {
          updated[updated.length - 1] = {
            role: "assistant",
            content: "Error: " + error.message,
            streaming: false,
            sources: [],
          };
        }
        return updated;
      });
    }

    setLoading(false);
    textareaRef.current?.focus();
  };

  //shift+enter handler for new line
  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  //UI
  return (
    <div className="app">
      <div className="sidebar">
        <h2>RAG Chat</h2>

        <label className={`upload ${uploading ? "disabled" : ""}`}>
          {uploading ? "Uploading..." : "Upload Files"}
          <input
            type="file"
            multiple
            onChange={handleUpload}
            disabled={uploading}
            hidden
          />
        </label>

        {uploadErrors.length > 0 && (
          <div className="upload-errors">
            {uploadErrors.map((f, idx) => (
              <div key={idx} className="upload-error">
                Failed: {f.file} — {f.error}
              </div>
            ))}
          </div>
        )}

        {deleteError && <div className="delete-error">{deleteError}</div>}

        <div className="files">
          {files.map((f, i) => (
            <div key={i} className="file">
              <span>{f}</span>
              <button
                onClick={() => removeFile(f)}
                disabled={deletingFile === f}
              >
                {deletingFile === f ? "..." : "X"}
              </button>
            </div>
          ))}
        </div>
      </div>

      <div className="chat">
        <div className="messages">
          {history.map((msg, i) => (
            <div key={i} className={`msg ${msg.role}`}>
              <div className="bubble">
                <pre>{msg.content}</pre>
                {msg.streaming && <span className="cursor">▍</span>}
              </div>

              {msg.sources?.length > 0 && (
                <div className="sources">
                  <h4>Sources</h4>
                  {msg.sources.map((s, idx) => (
                    <div key={idx} className="source">
                      <strong>{s.file}</strong>
                      <p>{s.preview}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}

          <div ref={bottomRef} />
        </div>

        <div className="input">
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask something..."
          />
        </div>
      </div>
    </div>
  );
}