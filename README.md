# DLSU Handbook Assistant — Full Stack App

React frontend + FastAPI backend + RAG pipeline

## Project Structure

```
dlsu-app/
├── backend/
│   ├── api.py                  ← FastAPI backend
│   ├── requirements.txt
│   ├── .env.example            ← copy to .env
│   ├── .gitignore
│   ├── student-handbook.pdf    ← add this yourself
│   ├── chunks.json             ← generated on first run
│   └── chunk_embeddings.npy   ← generated on first run
└── frontend/
    ├── src/
    │   ├── App.js
    │   ├── App.css
    │   └── index.js
    ├── public/
    │   └── index.html
    ├── package.json
    └── .env.example
```

---

## Local Setup

### Backend

```bash
cd backend

# 1. create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. set up environment variables
cp .env.example .env
# edit .env and add your GROQ_API_KEY

# 4. add your PDF
# copy student-handbook.pdf to the backend/ folder

# 5. start the server
uvicorn api:app --reload --port 8000
```

First run generates embeddings (~3 min on GPU, ~25 min on CPU).
Subsequent runs load from disk instantly.

### Frontend

```bash
cd frontend

# 1. install dependencies
npm install

# 2. set API URL
cp .env.example .env.local
# edit .env.local if needed (default points to localhost:8000)

# 3. start the app
npm start
```

Opens at http://localhost:3000

---

## Deploy to Production

### Backend → Render

1. Push backend/ to GitHub (skip chunk_embeddings.npy — too large)
2. Go to render.com → New Web Service
3. Connect your repo
4. Set:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`
5. Add environment variable: `GROQ_API_KEY=your_key`
6. Deploy — first deploy generates embeddings (~10 min)
7. Copy your Render URL (e.g. https://dlsu-api.onrender.com)

### Frontend → Vercel

1. Push frontend/ to GitHub
2. Go to vercel.com → New Project
3. Import your repo
4. Add environment variable:
   - `REACT_APP_API_URL=https://dlsu-api.onrender.com`
5. Deploy
6. Also update CORS in api.py:
   - Replace `https://your-app.vercel.app` with your actual Vercel URL

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | /health  | Check server status |
| POST   | /chat    | Send a question, get an answer |
| DELETE | /history | Clear conversation history |

### POST /chat

Request:
```json
{ "question": "What are the attendance rules?" }
```

Response:
```json
{
  "answer": "According to Section 8...",
  "sources": [
    { "chunk_id": 42, "score": 4.11, "preview": "8.1 Attendance..." }
  ],
  "confidence": "HIGH"
}
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18 |
| Backend | FastAPI + Python |
| Embeddings | BAAI/bge-base-en-v1.5 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-12-v2 |
| Keyword search | BM25Okapi |
| LLM | Groq (Llama 3.3 70B) |
| Backend hosting | Render |
| Frontend hosting | Vercel |
