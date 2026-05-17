import { useState, useEffect } from "react";
import axios from "axios";
import "./LandingPage.css";

const API = process.env.REACT_APP_API_URL || "http://localhost:8000";

export default function LandingPage({ onStart }) {
  const [stats, setStats] = useState(null);

  useEffect(() => {
    axios.get(`${API}/stats`)
      .then(r => setStats(r.data))
      .catch(() => {});
  }, []);

  return (
    <div className="landing">
      <div className="landing-card">
        <div className="landing-logo">La</div>
        <h1 className="landing-title">DLSU Handbook Assistant</h1>
        <p className="landing-subtitle">
          Your AI guide to the DLSU Student Handbook 2021-2025
        </p>

        <div className="landing-stats">
          <div className="stat-item">
            <div className="stat-number">
              {stats ? stats.total_visits.toLocaleString() : "—"}
            </div>
            <div className="stat-label">Total visits</div>
          </div>
          <div className="stat-divider" />
          <div className="stat-item">
            <div className="stat-number">
              {stats ? stats.total_questions.toLocaleString() : "—"}
            </div>
            <div className="stat-label">Questions answered</div>
          </div>
        </div>

        <button className="landing-btn" onClick={onStart}>
          Start chatting →
        </button>

        <p className="landing-disclaimer">
          This assistant is powered by AI and may occasionally give incorrect or
          incomplete answers. Always verify important information with the official
          DLSU Student Handbook or consult the SDFO and relevant offices directly.
        </p>
      </div>
    </div>
  );
}
