import { useState, useRef, useEffect } from "react";
import axios from "axios";
import "./App.css";

const API = process.env.REACT_APP_API_URL || "http://localhost:8000";

const SUGGESTIONS = [
  "What are the attendance rules?",
  "What are examples of grave offenses?",
  "What are the graduation requirements?",
  "What happens if I cheat on an exam?",
  "What are my rights as a DLSU student?",
  "What are the Latin honors requirements?",
];

function TypingIndicator() {
  return (
    <div className="message-row bot">
      <div className="avatar">La</div>
      <div className="bubble-wrapper">
        <div className="bubble bot">
          <div className="typing">
            <span /><span /><span />
          </div>
        </div>
      </div>
    </div>
  );
}

function Message({ msg, showSources }) {
  const confidenceColor = {
    HIGH: "#22c55e",
    MEDIUM: "#f59e0b",
    LOW: "#ef4444",
  };

  return (
    <div className={`message-row ${msg.role}`}>
      {msg.role === "bot" && <div className="avatar">La</div>}
      <div className="bubble-wrapper">
        <div className={`bubble ${msg.role}`}>
          {msg.text}
        </div>
        {showSources && msg.sources?.length > 0 && (
          <div className="sources">
            <span style={{ color: confidenceColor[msg.confidence] }}>
              ● {msg.confidence}
            </span>
            {" · "}
            {msg.sources.length} sources
            <div className="source-list">
              {msg.sources.map((s, i) => (
                <div key={i} className="source-item">
                  <span className="source-score">{s.score.toFixed(2)}</span>
                  {s.preview}...
                </div>
              ))}
            </div>
          </div>
        )}
        {msg.error && (
          <div className="error-msg">{msg.text}</div>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput]       = useState("");
  const [loading, setLoading]   = useState(false);
  const [showSources, setShowSources] = useState(false);
  const bottomRef = useRef(null);
  const inputRef  = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const send = async (question) => {
    const q = (question || input).trim();
    if (!q || loading) return;
    setInput("");
    setMessages(prev => [...prev, { role: "user", text: q }]);
    setLoading(true);

    try {
      const { data } = await axios.post(`${API}/chat`, { question: q });
      setMessages(prev => [...prev, {
        role: "bot",
        text: data.answer,
        sources: data.sources,
        confidence: data.confidence,
      }]);
    } catch (err) {
      const detail = err.response?.data?.detail || "Error connecting to server.";
      setMessages(prev => [...prev, {
        role: "bot",
        text: detail,
        sources: [],
        confidence: "LOW",
        error: true,
      }]);
    }

    setLoading(false);
    inputRef.current?.focus();
  };

  const clearChat = async () => {
    try { await axios.delete(`${API}/history`); } catch {}
    setMessages([]);
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-left">
          <div className="logo">La</div>
          <div>
            <div className="title">DLSU Handbook Assistant</div>
            <div className="subtitle">Student Handbook 2021-2025</div>
          </div>
        </div>
        <div className="header-right">
          <button
            className="sources-btn"
            onClick={() => setShowSources(s => !s)}
          >
            {showSources ? "Hide sources" : "View sources"}
          </button>
          <button className="clear-btn" onClick={clearChat}>
            Clear chat
          </button>
        </div>
      </header>

      <main className="messages">
        {messages.length === 0 && (
          <div className="welcome">
            <div className="welcome-logo">La</div>
            <h2>DLSU Handbook Assistant</h2>
            <p>Ask me anything about the DLSU Student Handbook 2021-2025</p>
            <div className="suggestions">
              {SUGGESTIONS.map(q => (
                <button key={q} className="suggestion" onClick={() => send(q)}>
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <Message key={i} msg={msg} showSources={showSources} />
        ))}

        {loading && <TypingIndicator />}
        <div ref={bottomRef} />
      </main>

      <footer className="input-area">
        <div className="input-row">
          <textarea
            ref={inputRef}
            className="input"
            value={input}
            onChange={e => {
              setInput(e.target.value);
              e.target.style.height = "auto";
              e.target.style.height = Math.min(e.target.scrollHeight, 160) + "px";
            }}
            onKeyDown={handleKey}
            placeholder="Ask about DLSU policies..."
            rows={1}
          />
          <button
            className="send-btn"
            onClick={() => send()}
            disabled={loading || !input.trim()}
          >
            ➤
          </button>
        </div>
        <div className="disclaimer">
          Responses based on DLSU Student Handbook 2021-2025. For official guidance, consult the SDFO or relevant office.
        </div>
      </footer>
    </div>
  );
}