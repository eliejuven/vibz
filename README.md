# vibz

MVP: generate 20–45s music from text / image / voice using MusicGen.
Baseline first, then LoRA fine-tune.

## Backend (local dev)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
uvicorn app:app --reload
```
Open: http://127.0.0.1:8000/docs