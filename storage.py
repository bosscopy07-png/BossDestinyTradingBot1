# storage.py
import os
import json
from datetime import datetime

DATA_FILE = os.getenv("DATA_FILE", "data.json")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def ensure_storage():
    if not os.path.exists(DATA_FILE):
        base = {
            "signals": [],
            "pnl": [],
            "challenge": {"balance": float(os.getenv("CHALLENGE_START", "10")), "wins": 0, "losses": 0, "history": []},
            "stats": {"total_signals": 0, "wins": 0, "losses": 0},
            "auto_scan": False
        }
        with open(DATA_FILE, "w") as f:
            json.dump(base, f, indent=2)

def load_data():
    ensure_storage()
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def save_data(d):
    tmp = DATA_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, DATA_FILE)

def record_pnl_screenshot(binary_bytes, now_str, user_id, caption):
    fname = f"{UPLOAD_DIR}/{now_str}_{user_id}.jpg"
    with open(fname, "wb") as f:
        f.write(binary_bytes)
    d = load_data()
    d["pnl"].append({"file": fname, "from": user_id, "time": now_str, "caption": caption, "linked": None})
    save_data(d)
    return fname
