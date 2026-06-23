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

}
