import { useState, useEffect, useRef } from "react";
import "./App.css";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

async function fetchFiles() {
  const res = await fetch(`${API}/files`);
  if (!res.ok) {
    throw new Error(`Server responded ${res.status}`);
  }
  const data = await res.json();
  return data.files || [];
}

export default function App() {
  const [files, setFiles] = useState([]);
  const [message, setMessage] = useState("");
  const [history, setHistory] = useState([]);
  const bottomRef = useRef(null);
  const textareaRef = useRef(null);
  const [loading, setLoading] = useState(false);

  const [uploading, setUploading] = useState(false);
  const [uploadErrors, setUploadErrors] = useState([]);

  const [deletingFile, setDeletingFile] = useState(null);
  const [deleteError, setDeleteError] = useState(null);
  const [filesError, setFilesError] = useState(null);

  const [report, setReport] = useState(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [reportError, setReportError] = useState(null);

  //File Handling
  const loadFiles = async () => {
    try {
      setFiles(await fetchFiles());
      setFilesError(null);
    } catch (err) {
      console.error("File load error:", err);
      setFilesError(err.message || "Could not reach the server");
    }
  };

  useEffect(() => {
    let cancelled = false;
    fetchFiles().then(
      (list) => {
        if (!cancelled) setFiles(list);
      },
      (err) => {
        console.error("File load error:", err);
        if (!cancelled) setFilesError(err.message || "Could not reach the server");
      }
    );
    return () => {
      cancelled = true;
    };
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

    const previousFiles = files;
    setFiles((prev) => prev.filter((f) => f !== fname));

    try {
      const res = await fetch(`${API}/files/${encodeURIComponent(fname)}`, {
        method: "DELETE",
      });

      if (!res.ok) {
        throw new Error(`Delete failed with status ${res.status}`);
      }

      await loadFiles();
    } catch (err) {
      console.error("Delete error:", err);
      setDeleteError(`Failed to delete "${fname}": ${err.message}`);
      setFiles(previousFiles);
    } finally {
      setDeletingFile(null);
    }
  };
  const getReport = async () => {
      setReportLoading(true);
      setReportError(null);
      try {
        const res = await fetch(`${API}/report/run`, { method: "POST" });
    
        if (!res.ok) {
          throw new Error(`Report generation failed with status ${res.status}`);
        }
    
        const data = await res.json();
    
        if (data.error) {
          throw new Error(data.error);
        }
    
        setReport(data);
      } catch (err) {
        console.error("Report error:", err);
        setReportError(err.message || "Could not generate report");
      } finally {
        setReportLoading(false);
      }
  };

  //Send handling
  const sendMessage = async () => {
    if (!message.trim() || loading) return;
    const userMsg = {
      role: "user",
      content: message.trim(),
    };
    // Prior turns only: the backend appends the new message itself, and
    // failed replies must not be fed back to the model as context.
    const priorHistory = history
      .filter((m) => !m.failed && m.content)
      .map(({ role, content }) => ({ role, content }));
    setHistory([
      ...history,
      userMsg,
      {
        role: "assistant",
        content: "",
        streaming: true,
        sources: [],
      },
    ]);
    setMessage("");
    setLoading(true);

    const updateLast = (patch) =>
      setHistory((prev) => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last?.role === "assistant") {
          updated[updated.length - 1] = {
            ...last,
            ...(typeof patch === "function" ? patch(last) : patch),
          };
        }
        return updated;
      });
    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: userMsg.content,
          history: priorHistory,
        }),
      });

      if (!res.ok || !res.body) {
        throw new Error(`Server responded ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      const handleLine = (line) => {
        if (!line.trim()) return;

        let data;
        try {
          data = JSON.parse(line);
        } catch {
          return;
        }

        if (data.type === "chunk") {
          updateLast((last) => ({ content: last.content + data.text }));
        } else if (data.type === "sources") {
          updateLast({ sources: data.sources });
        } else if (data.type === "error") {
          throw new Error(data.error || "Generation failed");
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) handleLine(line);
      }

      handleLine(buffer + decoder.decode());

      // Always clear the cursor, even if the stream ended without "done"
      updateLast({ streaming: false });
    } catch (error) {
      updateLast({
        content: "Error: " + error.message,
        streaming: false,
        sources: [],
        failed: true,
      });
    } finally {
      setLoading(false);
      textareaRef.current?.focus();
    }
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

        {filesError && (
          <div className="delete-error">
            Couldn't load files: {filesError}
          </div>
        )}

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
        <div className="reports">
          <button onClick={getReport} disabled={reportLoading}>
            {reportLoading ? "Running eval..." : "Get Report"}
          </button>
          {reportError && <div className="delete-error">{reportError}</div>}
          {report && (
            <div className="report-panel">
              <h4>Eval Report</h4>
              <div className="report-row">
                <span>Cases run</span>
                <span>{report.num_cases}</span>
              </div>
              <div className="report-row">
                <span>Retrieval recall</span>
                <span>{report.recall_at_10 != null ? `${(report.recall_at_10 * 100).toFixed(0)}%` : "—"}</span>
              </div>
              <div className="report-row">
                <span>Hallucination rate</span>
                <span>{report.hallucination_rate != null ? `${(report.hallucination_rate * 100).toFixed(0)}%` : "—"}</span>
              </div>
              <div className="report-row">
                <span>p50 latency</span>
                <span>{report.latency_p50_ms != null ? `${report.latency_p50_ms}ms` : "—"}</span>
              </div>
              <div className="report-row">
                <span>p95 latency</span>
                <span>{report.latency_p95_ms != null ? `${report.latency_p95_ms}ms` : "—"}</span>
              </div>
              <div className="report-row">
                <span>Cold start</span>
                <span>{report.cold_start_seconds != null ? `${report.cold_start_seconds}s` : "not captured"}</span>
              </div>
              <div className="report-timestamp">Generated {report.generated_at}</div>
            </div>
          )}
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