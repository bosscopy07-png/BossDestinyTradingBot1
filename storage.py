# storage.py
import os
import json
from datetime import datetime

STORAGE_DIR = os.getenv("STORAGE_DIR", "storage")
DATA_FILE = os.path.join(STORAGE_DIR, "data.json")
PNL_DIR = os.path.join(STORAGE_DIR, "pnl")

DEFAULT_DATA = {
    "signals": [],
    "pnl": [],
    "challenge": {"balance": float(os.getenv("CHALLENGE_START", "100.0")), "wins": 0, "losses": 0},
    "stats": {}
}

def ensure_storage():
    """Create storage dirs and data file if missing."""
    os.makedirs(STORAGE_DIR, exist_ok=True)
    os.makedirs(PNL_DIR, exist_ok=True)
    if not os.path.exists(DATA_FILE):
        save_data(DEFAULT_DATA)

def load_data():
    """Load JSON data atomically (returns dict)."""
    ensure_storage()
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # recreate with defaults on failure
        save_data(DEFAULT_DATA)
        return DEFAULT_DATA.copy()

def save_data(d):
    """Save dict to DATA_FILE atomically."""
    ensure_storage()
    tmp = DATA_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
    os.replace(tmp, DATA_FILE)

def record_pnl_screenshot(data_bytes, fname=None, from_user=None, caption=None):
    """
    Save screenshot bytes to PNL_DIR and register in data.json.
    Returns saved filename.
    """
    ensure_storage()
    if fname is None:
        fname = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    # keep extension png by default
    if not (fname.lower().endswith(".png") or fname.lower().endswith(".jpg") or fname.lower().endswith(".jpeg")):
        fname = f"{fname}.png"
    path = os.path.join(PNL_DIR, fname)
    with open(path, "wb") as f:
        f.write(data_bytes)
    # register
    d = load_data()
    item = {
        "file": fname,
        "path": path,
        "from": from_user,
        "caption": caption,
        "time": datetime.utcnow().isoformat(),
        "linked": None
    }
    d.setdefault("pnl", []).append(item)
    save_data(d)
    return fname
