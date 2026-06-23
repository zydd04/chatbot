import { useState, useEffect, useRef  } from "react";
const API = "http://127.0.0.1:8000";

export default function App() {
    const [files, setFiles] = useState([]);
    const [message, setMessage] = useState([]);
    const [history, setHistory] = useState([]);
    const bottomRef = useRef(null);
    const [loading, setLoading] = useState(false);

    //File Handling
    const loadFiles = async () => {
        try {
            const res = await(fetch(`${API}/docs`))
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

    //Upload handling
    const handleUpload = async(e) => {
        const file = e.target.docs[0];
        if (!file) return; //empty
    const formData = new FormData();
    formData.append("file", file);
    await fetch(`${API}/upload`, {
      method: "POST",
      body: formData,
    });
    loadFiles();
        e.target.value = "";
    };

    //Delete handling
    const deleteFile = async (fname) => {
        await fetch(`${API}/docs/${filename}`, {
        method: "DELETE",
        });
        loadFiles(); //reload
    };

    //Send handling
    const sendMessage = async () => {
        if (!message.trim() || loading) return; //can't send while loading or empty message
        const userMsg = {
        role: "user",
        content: message.trim(),
        };
        const updatedHistory = [...history, userMsg]; //update history
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
            //read
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
                
                //show chunks
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
    
} 
