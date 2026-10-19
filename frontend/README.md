# Agentic RAG React Dashboard

React + Vite frontend for the existing FastAPI backend.

## Requirements
Install Node.js 20.19+ or 22.12+.

## Run
```powershell
npm install
npm run dev
```
Open `http://localhost:5173`.

## Required FastAPI CORS configuration
In `src/api.py`, add this import alongside the other imports:
```python
from fastapi.middleware.cors import CORSMiddleware
```
Immediately after your existing `app = FastAPI(...)` line, add:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
Keep your existing routes and backend code. Restart Uvicorn after this change.

The dashboard uses `GET /health` and `POST /api/query`. Configure another API base URL in `.env.local` with `VITE_API_BASE_URL=...` if needed. Never put backend API keys in frontend environment variables.
