# Backend

Run from the repository root with:

```powershell
python -m uvicorn server.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Install dependencies with `python -m pip install -r server/requirements.txt`.
Runtime data is stored under `server/data/`, and the SQLite database is
`server/events.db`.
