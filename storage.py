# storage.py
import os
import json
from pathlib import Path
from datetime import datetime

DATA_PATH = Path(os.getenv("DATA_PATH", "data"))
DATA_PATH.mkdir(parents=True, exist_ok=True)
DB_FILE = DATA_PATH / "state.json"

DEFAULT = {
    "signals": [],
    "pnl": [],
    "challenge": {"balance": float(os.getenv("CHALLENGE_START", "100.0")), "wins": 0, "losses": 0},
    "stats": {}
}

def ensure_storage():
    if not DB_FILE.exists():
        with open(DB_FILE, "w") as f:
            json.dump(DEFAULT, f, indent=2)

def load_data():
    try:
        if not DB_FILE.exists():
            ensure_storage()
        with open(DB_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return DEFAULT.copy()

def save_data(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)

def record_pnl_screenshot(bytes_data, fname_ts, from_user_id, caption=None):
    # saves screenshot as file and records in DB
    fname = DATA_PATH / f"pnl_{fname_ts}.png"
    with open(fname, "wb") as f:
        f.write(bytes_data)
    d = load_data()
    rec = {"file": str(fname), "from": from_user_id, "caption": caption, "ts": datetime.utcnow().isoformat(), "linked": None}
    d.setdefault("pnl", []).append(rec)
    save_data(d)
    return str(fname)
