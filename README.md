# Backend

```powershell
.\.venv\Scripts\Activate
python -m uvicorn server.app.main:app --host 127.0.0.1 --port 8000 --reload
```

# Frontend

```powershell
cd frontend
npm install
npm run dev
```
