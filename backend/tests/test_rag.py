"""
RAG pipeline unit tests
"""
import pytest
from unittest.mock import patch, MagicMock
import sys
import os
import numpy as np

# mock heavy deps
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["rank_bm25"] = MagicMock()
sys.modules["pypdf"] = MagicMock()
sys.modules["groq"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["google"] = MagicMock()
sys.modules["google.genai"] = MagicMock()
sys.modules["resend"] = MagicMock()
sys.modules["sklearn"] = MagicMock()
sys.modules["sklearn.metrics"] = MagicMock()
sys.modules["sklearn.metrics.pairwise"] = MagicMock()

os.environ.setdefault("GROQ_API_KEY", "test_key")

with patch("numpy.load", return_value=np.zeros((10, 768))), \
     patch("builtins.open", MagicMock()), \
     patch("json.load", return_value=["chunk " + str(i) for i in range(10)]), \
     patch("os.path.exists", return_value=True):
    from api import build_prompt, chunk_pdf


class TestBuildPrompt:
    def test_prompt_contains_context(self):
        chunks = [{"chunk_id": 0, "rerank_score": 0.03, "text": "Section 8: Attendance rules"}]
        prompt = build_prompt(chunks)
        assert "Section 8" in prompt
        assert "Attendance rules" in prompt

    def test_prompt_has_system_instructions(self):
        chunks = [{"chunk_id": 0, "rerank_score": 0.03, "text": "test"}]
        prompt = build_prompt(chunks)
        assert "DLSU" in prompt
        assert "handbook" in prompt.lower()

    def test_prompt_has_not_human_instruction(self):
        chunks = [{"chunk_id": 0, "rerank_score": 0.03, "text": "test"}]
        prompt = build_prompt(chunks)
        assert "not a human" in prompt

    def test_prompt_has_cite_instruction(self):
        chunks = [{"chunk_id": 0, "rerank_score": 0.03, "text": "test"}]
        prompt = build_prompt(chunks)
        assert "section number" in prompt.lower() or "cite" in prompt.lower()

    def test_prompt_has_not_found_instruction(self):
        chunks = [{"chunk_id": 0, "rerank_score": 0.03, "text": "test"}]
        prompt = build_prompt(chunks)
        assert "could not find" in prompt.lower()

    def test_prompt_multiple_chunks(self):
        chunks = [
            {"chunk_id": 0, "rerank_score": 0.03, "text": "Section 1 content"},
            {"chunk_id": 1, "rerank_score": 0.02, "text": "Section 2 content"},
        ]
        prompt = build_prompt(chunks)
        assert "Section 1" in prompt
        assert "Section 2" in prompt
        assert "[Section 1]" in prompt
        assert "[Section 2]" in prompt


class TestInputSanitization:
    """Test that banned phrases are caught"""

    BANNED = [
        "ignore previous",
        "system prompt",
        "jailbreak",
        "forget everything",
        "ignore instructions",
        "disregard",
    ]

    def test_all_banned_phrases_detected(self):
        from fastapi.testclient import TestClient
        from api import app
        c = TestClient(app)
        for phrase in self.BANNED:
            resp = c.post("/chat", json={"question": phrase})
            assert resp.status_code == 422, f"Expected 422 for phrase: '{phrase}'"

    def test_normal_question_passes(self):
        from fastapi.testclient import TestClient
        from api import app
        c = TestClient(app)
        with patch("api.call_llm", return_value=("answer", 50)), \
             patch("api.retrieve", return_value=[
                 {"chunk_id": 0, "rerank_score": 0.03, "text": "test"}
             ]):
            resp = c.post("/chat", json={"question": "What are the attendance rules?"})
            assert resp.status_code == 200
