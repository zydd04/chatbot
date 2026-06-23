import { useState, useEffect, useRef  } from "react";
const API = "http://127.0.0.1:8000";

export default function App() {
    const [files, setFiles] = useState([]);
    const [message, setMessage] = useState([]);
    const [history, setHistory] = useState([]);

    const bottomRef = useRef(null);
    const textareaRef = useRef(null);

    //File Handling
    const loadFile = asyn () => {
        try {
            const res = await(fetch(`${API}/docs`))
            const data = await res.json();
            setFiles(data.files || []);
        } catch (err) {
        console.error("File load error:", err);
        }
    };
}
