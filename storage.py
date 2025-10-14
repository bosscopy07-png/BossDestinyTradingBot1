import os
import json
import threading
from datetime import datetime

DATA_FILE = os.getenv("DATA_FILE", "data.json")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
BACKUP_DIR = os.getenv("BACKUP_DIR", "backups")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)

# Thread-safe lock for reading/writing data
_data_lock = threading.Lock()

def _default_data():
    return {
        "signals": [],
        "pnl": [],
        "challenge": {
            "balance": float(os.getenv("CHALLENGE_START", "10")),
            "wins": 0,
            "losses": 0,
            "history": []
        },
        "stats": {"total_signals": 0, "wins": 0, "losses": 0},
        "auto_scan": False
    }

def ensure_storage():
    """Ensure the main data file exists."""
    if not os.path.exists(DATA_FILE):
        save_data(_default_data())

def load_data():
    """Thread-safe data load."""
    ensure_storage()
    with _data_lock:
        try:
            with open(DATA_FILE, "r") as f:
                return json.load(f)
        except Exception:
            # fallback: recreate data file
            save_data(_default_data())
            return _default_data()

def save_data(d):
    """Thread-safe save with atomic replace and optional backup."""
    with _data_lock:
        tmp_file = DATA_FILE + ".tmp"
        with open(tmp_file, "w") as f:
            json.dump(d, f, indent=2)
        os.replace(tmp_file, DATA_FILE)

        # optional backup every save
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(BACKUP_DIR, f"data_backup_{timestamp}.json")
        with open(backup_file, "w") as f:
            json.dump(d, f, indent=2)

def record_signal(signal_data):
    """Add a new signal to storage."""
    d = load_data()
    d["signals"].append(signal_data)
    d["stats"]["total_signals"] += 1
    save_data(d)

def record_pnl_screenshot(binary_bytes, user_id, caption=""):
    """Save PnL screenshot and record metadata."""
    now_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(UPLOAD_DIR, f"{now_str}_{user_id}.jpg")
    with open(fname, "wb") as f:
        f.write(binary_bytes)

    d = load_data()
    d["pnl"].append({
        "file": fname,
        "from": user_id,
        "time": now_str,
        "caption": caption,
        "linked": None
    })
    save_data(d)
    return fname

def update_challenge(balance=None, win_loss=None):
    """Update challenge stats."""
    d = load_data()
    challenge = d.get("challenge", {})
    if balance is not None:
        challenge["balance"] = balance
    if win_loss:
        w, l = win_loss
        challenge["wins"] += w
        challenge["losses"] += l
    d["challenge"] = challenge
    save_data(d)
    return challenge

def update_stats(wins=0, losses=0):
    """Update global stats."""
    d = load_data()
    d["stats"]["wins"] += wins
    d["stats"]["losses"] += losses
    save_data(d)
    return d["stats"]
