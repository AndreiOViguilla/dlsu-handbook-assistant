import { useState, useRef, useEffect } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
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

function FeedbackModal({ onClose, onSubmit, prefillMessage }) {
  const [issue, setIssue] = useState(prefillMessage || "");
  const [sent, setSent]   = useState(false);

  const handleSubmit = async () => {
    if (!issue.trim()) return;
    await onSubmit(issue);
    setSent(true);
    setTimeout(onClose, 2000);
  };

  return (
    <div className="modal-overlay">
      <div className="modal">
        {sent ? (
          <div className="modal-sent">
            <div className="modal-sent-icon">✓</div>
            <p>Thank you! Your feedback has been sent.</p>
          </div>
        ) : (
          <>
            <div className="modal-header">
              <h3>Report an issue</h3>
              <button className="modal-close" onClick={onClose}>✕</button>
            </div>
            <p className="modal-desc">
              Describe what went wrong or what answer you were expecting.
            </p>
            <textarea
              className="modal-input"
              placeholder="Describe the issue..."
              value={issue}
              onChange={e => setIssue(e.target.value)}
              rows={4}
            />
            <div className="modal-actions">
              <button className="modal-cancel" onClick={onClose}>Cancel</button>
              <button className="modal-submit" onClick={handleSubmit}>Send report</button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function CheckFeedbackModal({ onClose, onSubmit }) {
  const [issue, setIssue] = useState("");
  const [sent, setSent]   = useState(false);

  const handleSubmit = async () => {
    if (!issue.trim()) return;
    await onSubmit(issue);
    setSent(true);
    setTimeout(onClose, 2000);
  };

  return (
    <div className="modal-overlay">
      <div className="modal">
        {sent ? (
          <div className="modal-sent">
            <div className="modal-sent-icon">✓</div>
            <p>Thank you! Your feedback has been sent.</p>
          </div>
        ) : (
          <>
            <div className="modal-header">
              <h3>Having trouble?</h3>
              <button className="modal-close" onClick={onClose}>✕</button>
            </div>
            <p className="modal-desc">
              It looks like you have asked a few questions. Is the bot giving you the answers you need?
            </p>
            <textarea
              className="modal-input"
              placeholder="Describe the issue or what answer you were expecting..."
              value={issue}
              onChange={e => setIssue(e.target.value)}
              rows={4}
            />
            <div className="modal-actions">
              <button className="modal-cancel" onClick={onClose}>No issue</button>
              <button className="modal-submit" onClick={handleSubmit}>Send feedback</button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function Message({ msg, showSources, onReport }) {
  const [copied, setCopied] = useState(false);

  const confidenceColor = {
    HIGH: "#22c55e",
    MEDIUM: "#f59e0b",
    LOW: "#ef4444",
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(msg.text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className={`message-row ${msg.role}`}>
      {msg.role === "bot" && <div className="avatar">La</div>}
      <div className="bubble-wrapper">
        <div className={`bubble ${msg.role}`}>
          {msg.role === "bot" ? (
            <ReactMarkdown>{msg.text}</ReactMarkdown>
          ) : (
            msg.text
          )}
        </div>

        {msg.role === "bot" && !msg.error && (
          <div className="message-actions">
            <button
              className="action-btn"
              onClick={handleCopy}
              title={copied ? "Copied!" : "Copy response"}
            >
              {copied ? (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="20 6 9 17 4 12"/>
                </svg>
              ) : (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                  <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                </svg>
              )}
            </button>
            <button
              className="action-btn"
              onClick={() => onReport(msg.text)}
              title="Report issue"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/>
                <line x1="4" y1="22" x2="4" y2="15"/>
              </svg>
            </button>
          </div>
        )}

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
  const [messages, setMessages]             = useState([]);
  const [input, setInput]                   = useState("");
  const [loading, setLoading]               = useState(false);
  const [showSources, setShowSources]       = useState(false);
  const [showCheckFeedback, setShowCheckFeedback] = useState(false);
  const [reportModal, setReportModal]       = useState(null); // holds msg text
  const [questionCount, setQuestionCount]   = useState(0);
  const bottomRef = useRef(null);
  const inputRef  = useRef(null);
  const abortRef  = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    if (questionCount === 2) {
      setShowCheckFeedback(true);
    }
  }, [questionCount]);

  const stop = () => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    setLoading(false);
    inputRef.current?.focus();
  };

const send = async (question) => {
    const q = (question || input).trim();
    if (!q || loading) return;
    setInput("");
    // reset textarea height back to normal
    if (inputRef.current) {
      inputRef.current.style.height = "auto";
    }
    setMessages(prev => [...prev, { role: "user", text: q }]);
    setLoading(true);

    abortRef.current = new AbortController();

    try {
      const { data } = await axios.post(
        `${API}/chat`,
        { question: q },
        { signal: abortRef.current.signal }
      );
      setMessages(prev => [...prev, {
        role: "bot",
        text: data.answer,
        sources: data.sources,
        confidence: data.confidence,
      }]);
      setQuestionCount(prev => prev + 1);
    } catch (err) {
      if (axios.isCancel(err) || err.name === "CanceledError") {
        // user stopped
      } else {
        const detail = typeof err.response?.data?.detail === "string"
          ? err.response.data.detail
          : "Error connecting to server.";
        setMessages(prev => [...prev, {
          role: "bot",
          text: detail,
          sources: [],
          confidence: "LOW",
          error: true,
        }]);
      }
    }

    setLoading(false);
    abortRef.current = null;
    inputRef.current?.focus();
  };

  const submitFeedback = async (issue, reportedMessage) => {
    try {
      const conversation = reportedMessage
        ? [{ role: "bot", text: reportedMessage }]
        : messages.slice(-6).map(m => ({ role: m.role, text: m.text }));

      await axios.post(`${API}/feedback`, { issue, conversation });
    } catch (err) {
      console.error("Feedback error:", err);
    }
  };

  const clearChat = async () => {
    try { await axios.delete(`${API}/history`); } catch {}
    setMessages([]);
    setQuestionCount(0);
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div className="app">
      {showCheckFeedback && (
        <CheckFeedbackModal
          onClose={() => setShowCheckFeedback(false)}
          onSubmit={(issue) => submitFeedback(issue, null)}
        />
      )}

      {reportModal !== null && (
        <FeedbackModal
          onClose={() => setReportModal(null)}
          onSubmit={(issue) => submitFeedback(issue, reportModal)}
          prefillMessage=""
        />
      )}

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
          <Message
            key={i}
            msg={msg}
            showSources={showSources}
            onReport={(msgText) => setReportModal(msgText)}
          />
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
            disabled={loading}
          />
          {loading ? (
            <button className="stop-btn" onClick={stop} title="Stop generating">
              &#9632;
            </button>
          ) : (
            <button
              className="send-btn"
              onClick={() => send()}
              disabled={!input.trim()}
            >
              ➤
            </button>
          )}
        </div>
        <div className="disclaimer">
          Responses based on DLSU Student Handbook 2021-2025. For official guidance, consult the SDFO or relevant office.
        </div>
      </footer>
    </div>
  );
}