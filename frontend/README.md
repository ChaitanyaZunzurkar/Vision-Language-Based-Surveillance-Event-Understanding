# Existing surveillance dashboard

This directory contains the existing browser dashboard served by FastAPI.
Start the backend from the repository root:

```powershell
python -m uvicorn server.app.main:app --reload
```

Then open <http://127.0.0.1:8000/>. The dashboard communicates with the
backend through `/api` and media is served from `/static`.
