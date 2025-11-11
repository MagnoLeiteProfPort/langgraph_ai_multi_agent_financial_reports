# Run FastAPI app with reload
$env:PYTHONUNBUFFERED=1
uvicorn src.app.main:app --host 127.0.0.1 --port 8787 --reload --access-log
