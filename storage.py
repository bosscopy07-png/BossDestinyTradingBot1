# storage.py
"""
Persistent storage layer for Destiny Trading Empire.
Handles JSON data, PnL screenshots, and challenge tracking with atomic operations.
"""

import os
import json
import logging
import shutil
import tempfile
import fcntl
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from contextlib import contextmanager

# Configure logging
logger = logging.getLogger("storage")
logger.setLevel(logging.INFO)

# ----- Configuration -----
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "storage"))
DATA_FILE = STORAGE_DIR / "data.json"
PNL_DIR = STORAGE_DIR / "pnl"
BACKUP_DIR = STORAGE_DIR / "backups"

# Atomic write settings
MAX_BACKUPS = 10

# Default data structure
DEFAULT_DATA = {
    "version": 1,
    "signals": [],
    "pnl": [],
    "challenge": {
        "balance": float(os.getenv("CHALLENGE_START", "100.0")),
        "start_balance": float(os.getenv("CHALLENGE_START", "100.0")),
        "wins": 0,
        "losses": 0,
        "total_trades": 0,
        "win_rate": 0.0,
        "current_streak": 0,
        "max_drawdown": 0.0
    },
    "stats": {
        "total_signals": 0,
        "total_pnl_uploads": 0,
        "last_updated": None
    },
    "settings": {}
}


@dataclass
class PnLRecord:
    """Structured PnL record."""
    file: str
    path: str
    from_user: Optional[int]
    caption: Optional[str]
    timestamp: str
    linked_signal: Optional[str] = None
    pnl_amount: Optional[float] = None
    pnl_percent: Optional[float] = None
    verified: bool = False


@dataclass
class SignalRecord:
    """Structured signal record."""
    signal_id: str
    symbol: str
    interval: str
    signal_type: str
    entry: Optional[float]
    sl: Optional[float]
    tp1: Optional[float]
    confidence: float
    timestamp: str
    user_id: Optional[int] = None
    auto: bool = False
    status: str = "open"  # open, tp1, sl, closed
    pnl_linked: Optional[str] = None


@dataclass
class ChallengeStats:
    """Challenge tracking statistics."""
    balance: float
    start_balance: float
    wins: int
    losses: int
    total_trades: int
    win_rate: float
    current_streak: int
    max_drawdown: float


# ----- File Locking -----
@contextmanager
def _file_lock(filepath: Path, exclusive: bool = True):
    """
    Context manager for file locking (Unix/Linux).
    Prevents race conditions during read/write operations.
    """
    lock_file = filepath.with_suffix(filepath.suffix + ".lock")
    
    try:
        # Create lock file if needed
        lock_file.touch(exist_ok=True)
        
        with open(lock_file, "r+") as f:
            if exclusive:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            yield
            if exclusive:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        logger.warning(f"File locking failed (non-critical): {e}")
        yield  # Continue without lock


# ----- Initialization -----
def ensure_storage() -> None:
    """
    Create storage directories and initialize data file if missing.
    Thread-safe with locking.
    """
    try:
        # Create directories
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        PNL_DIR.mkdir(parents=True, exist_ok=True)
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        
        # Initialize data file if missing
        if not DATA_FILE.exists():
            with _file_lock(DATA_FILE, exclusive=True):
                _atomic_write_json(DEFAULT_DATA, DATA_FILE)
            logger.info(f"Created new data file at {DATA_FILE}")
        
    except Exception as e:
        logger.error(f"Failed to ensure storage: {e}")
        raise


# ----- Atomic Operations -----
def _atomic_write_json(data: Dict[str, Any], filepath: Path) -> None:
    """
    Write JSON atomically using temp file + rename.
    Prevents corruption if process crashes during write.
    """
    # Use temp file in same directory for atomic rename
    temp_fd, temp_path = tempfile.mkstemp(
        dir=filepath.parent,
        prefix=f".{filepath.stem}_",
        suffix=".tmp"
    )
    
    try:
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            f.flush()
            os.fsync(f.fileno())  # Ensure written to disk
        
        # Atomic replace
        os.replace(temp_path, filepath)
        
    except Exception as e:
        # Clean up temp file on failure
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise e


def _create_backup() -> Optional[Path]:
    """Create timestamped backup of data file."""
    try:
        if not DATA_FILE.exists():
            return None
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_path = BACKUP_DIR / f"data_{timestamp}.json"
        
        shutil.copy2(DATA_FILE, backup_path)
        
        # Clean old backups
        backups = sorted(BACKUP_DIR.glob("data_*.json"))
        if len(backups) > MAX_BACKUPS:
            for old in backups[:-MAX_BACKUPS]:
                old.unlink()
                logger.debug(f"Removed old backup: {old}")
        
        return backup_path
        
    except Exception as e:
        logger.warning(f"Backup creation failed: {e}")
        return None


# ----- Data Access -----
def load_data() -> Dict[str, Any]:
    """
    Load JSON data with automatic recovery from corruption.
    Thread-safe with shared lock.
    """
    ensure_storage()
    
    with _file_lock(DATA_FILE, exclusive=False):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # Migration: add missing fields
                data = _migrate_data(data)
                return data
                
        except json.JSONDecodeError as e:
            logger.error(f"Corrupted data file: {e}")
            # Try to recover from backup
            backup = _get_latest_backup()
            if backup:
                logger.info(f"Attempting recovery from {backup}")
                try:
                    with open(backup, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as be:
                    logger.error(f"Backup also corrupted: {be}")
            
            # Reset to defaults
            logger.warning("Resetting to default data")
            save_data(DEFAULT_DATA.copy())
            return DEFAULT_DATA.copy()
            
        except Exception as e:
            logger.error(f"Load failed: {e}")
            return DEFAULT_DATA.copy()


def save_data(data: Dict[str, Any], backup: bool = True) -> None:
    """
    Save data atomically with optional backup creation.
    Thread-safe with exclusive lock.
    """
    ensure_storage()
    
    # Update metadata
    data["stats"]["last_updated"] = datetime.utcnow().isoformat()
    
    with _file_lock(DATA_FILE, exclusive=True):
        if backup:
            _create_backup()
        _atomic_write_json(data, DATA_FILE)
    
    logger.debug(f"Data saved: {len(data.get('signals', []))} signals, "
                f"{len(data.get('pnl', []))} PnL records")


def _migrate_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate old data formats to current version."""
    # Ensure version field
    if "version" not in data:
        data["version"] = 1
    
    # Ensure challenge has all fields
    if "challenge" in data:
        challenge = data["challenge"]
        defaults = DEFAULT_DATA["challenge"]
        for key, val in defaults.items():
            if key not in challenge:
                challenge[key] = val
        
        # Calculate derived fields
        total = challenge.get("wins", 0) + challenge.get("losses", 0)
        if total > 0:
            challenge["win_rate"] = round(challenge.get("wins", 0) / total, 4)
            challenge["total_trades"] = total
    
    # Ensure stats has all fields
    if "stats" in data:
        stats = data["stats"]
        defaults = DEFAULT_DATA["stats"]
        for key, val in defaults.items():
            if key not in stats:
                stats[key] = val
    
    return data


def _get_latest_backup() -> Optional[Path]:
    """Find most recent backup file."""
    try:
        backups = sorted(BACKUP_DIR.glob("data_*.json"))
        return backups[-1] if backups else None
    except Exception:
        return None


# ----- Signal Management -----
def record_signal(signal_data: Dict[str, Any]) -> str:
    """
    Record a new trading signal with generated ID.
    Returns the signal ID.
    """
    data = load_data()
    
    # Generate ID
    signal_id = f"S{int(datetime.utcnow().timestamp() * 1000)}"
    
    # Create structured record
    record = SignalRecord(
        signal_id=signal_id,
        symbol=signal_data.get("symbol", "UNKNOWN"),
        interval=signal_data.get("interval", "1h"),
        signal_type=signal_data.get("signal", "HOLD"),
        entry=signal_data.get("entry"),
        sl=signal_data.get("sl"),
        tp1=signal_data.get("tp1"),
        confidence=signal_data.get("confidence", 0.0),
        timestamp=datetime.utcnow().isoformat(),
        user_id=signal_data.get("user_id"),
        auto=signal_data.get("auto", False),
        status="open"
    )
    
    # Add to storage
    signals = data.setdefault("signals", [])
    signals.append(asdict(record))
    
    # Update stats
    data["stats"]["total_signals"] = data["stats"].get("total_signals", 0) + 1
    
    save_data(data)
    logger.info(f"Recorded signal {signal_id} for {record.symbol}")
    
    return signal_id


def get_signal_history(limit: int = 50, status: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieve signal history with optional filtering.
    """
    data = load_data()
    signals = data.get("signals", [])
    
    # Sort by timestamp (newest first)
    signals = sorted(signals, key=lambda x: x.get("timestamp", ""), reverse=True)
    
    # Filter by status if specified
    if status:
        signals = [s for s in signals if s.get("status") == status]
    
    return signals[:limit]


def update_signal_status(signal_id: str, status: str, pnl_linked: Optional[str] = None) -> bool:
    """
    Update signal status (e.g., when TP or SL hit).
    """
    data = load_data()
    signals = data.get("signals", [])
    
    for signal in signals:
        if signal.get("signal_id") == signal_id:
            signal["status"] = status
            if pnl_linked:
                signal["pnl_linked"] = pnl_linked
            save_data(data)
            logger.info(f"Updated signal {signal_id} status to {status}")
            return True
    
    logger.warning(f"Signal {signal_id} not found for status update")
    return False


def link_pnl_to_signal(signal_id: str, pnl_file: str) -> bool:
    """Link a PnL screenshot to a specific signal."""
    data = load_data()
    
    # Find signal
    signals = data.get("signals", [])
    signal = next((s for s in signals if s.get("signal_id") == signal_id), None)
    
    if not signal:
        logger.warning(f"Cannot link PnL: signal {signal_id} not found")
        return False
    
    # Find PnL record
    pnl_records = data.get("pnl", [])
    pnl_record = next((p for p in pnl_records if p.get("file") == pnl_file), None)
    
    if not pnl_record:
        logger.warning(f"Cannot link PnL: file {pnl_file} not found")
        return False
    
    # Update both records
    signal["pnl_linked"] = pnl_file
    signal["status"] = "closed"
    pnl_record["linked_signal"] = signal_id
    
    # Update challenge stats if PnL amount available
    if pnl_record.get("pnl_amount"):
        _update_challenge_with_pnl(data, pnl_record["pnl_amount"])
    
    save_data(data)
    logger.info(f"Linked PnL {pnl_file} to signal {signal_id}")
    return True


# ----- PnL Management -----
def record_pnl_screenshot(
    data_bytes: bytes,
    filename: Optional[str] = None,
    from_user: Optional[int] = None,
    caption: Optional[str] = None,
    extract_pnl: bool = False
) -> str:
    """
    Save PnL screenshot and register in database.
    Optionally extract PnL amount from caption.
    
    Returns saved filename.
    """
    ensure_storage()
    
    # Generate filename
    if filename is None:
        filename = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    # Ensure proper extension
    filename = Path(filename).stem
    ext = ".png"  # Default
    
    # Try to detect format from bytes
    if data_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        ext = ".png"
    elif data_bytes[:2] == b'\xff\xd8':
        ext = ".jpg"
    
    full_filename = f"{filename}{ext}"
    filepath = PNL_DIR / full_filename
    
    # Save file
    with open(filepath, "wb") as f:
        f.write(data_bytes)
    
    # Extract PnL from caption if requested
    pnl_amount = None
    pnl_percent = None
    if extract_pnl and caption:
        pnl_amount, pnl_percent = _extract_pnl_from_caption(caption)
    
    # Create record
    record = PnLRecord(
        file=full_filename,
        path=str(filepath),
        from_user=from_user,
        caption=caption,
        timestamp=datetime.utcnow().isoformat(),
        linked_signal=None,
        pnl_amount=pnl_amount,
        pnl_percent=pnl_percent,
        verified=False
    )
    
    # Save to database
    data = load_data()
    data.setdefault("pnl", []).append(asdict(record))
    data["stats"]["total_pnl_uploads"] = data["stats"].get("total_pnl_uploads", 0) + 1
    
    save_data(data)
    logger.info(f"Saved PnL screenshot: {full_filename} "
                f"(user={from_user}, extracted_pnl={pnl_amount})")
    
    return full_filename


def _extract_pnl_from_caption(caption: str) -> tuple:
    """
    Attempt to extract PnL amount and percentage from caption.
    Returns (amount, percent) or (None, None).
    """
    import re
    
    # Look for patterns like: +$123.45, -50%, PnL: 25.5, etc.
    patterns = [
        r'[Pp][Nn][Ll][:\s]*([+-]?\d+\.?\d*)',  # PnL: 123.45
        r'([+-]?\d+\.?\d*)\s*[%％]',             # 12.5%
        r'[+-]?\$\s*(\d+\.?\d*)',                # $123.45
        r'([+-]?\d+\.?\d*)\s*(?:USD|USDT)',     # 123.45 USD
    ]
    
    amount = None
    percent = None
    
    for pattern in patterns:
        matches = re.findall(pattern, caption)
        for match in matches:
            try:
                val = float(match)
                if -1000 < val < 1000:  # Sanity check
                    if abs(val) < 100 and percent is None:
                        percent = val
                    elif amount is None:
                        amount = val
            except ValueError:
                continue
    
    return amount, percent


def get_pnl_history(limit: int = 20, linked_only: bool = False) -> List[Dict[str, Any]]:
    """Retrieve PnL screenshot history."""
    data = load_data()
    records = data.get("pnl", [])
    
    # Sort by timestamp
    records = sorted(records, key=lambda x: x.get("timestamp", ""), reverse=True)
    
    if linked_only:
        records = [r for r in records if r.get("linked_signal")]
    
    return records[:limit]


# ----- Challenge Management -----
def get_challenge_stats() -> ChallengeStats:
    """Get current challenge statistics."""
    data = load_data()
    challenge = data.get("challenge", DEFAULT_DATA["challenge"])
    return ChallengeStats(**challenge)


def update_challenge_balance(new_balance: float) -> None:
    """Update challenge balance and recalculate stats."""
    data = load_data()
    challenge = data.setdefault("challenge", DEFAULT_DATA["challenge"].copy())
    
    old_balance = challenge.get("balance", 100.0)
    challenge["balance"] = new_balance
    
    # Calculate drawdown
    if new_balance < old_balance:
        drawdown = (old_balance - new_balance) / challenge.get("start_balance", 100.0)
        current_max = challenge.get("max_drawdown", 0.0)
        challenge["max_drawdown"] = max(current_max, drawdown)
    
    save_data(data)
    logger.info(f"Challenge balance updated: {old_balance} -> {new_balance}")


def _update_challenge_with_pnl(data: Dict[str, Any], pnl_amount: float) -> None:
    """Internal: update challenge stats with PnL result."""
    challenge = data.setdefault("challenge", DEFAULT_DATA["challenge"].copy())
    
    # Determine win/loss
    if pnl_amount > 0:
        challenge["wins"] = challenge.get("wins", 0) + 1
        challenge["current_streak"] = max(0, challenge.get("current_streak", 0)) + 1
    else:
        challenge["losses"] = challenge.get("losses", 0) + 1
        challenge["current_streak"] = min(0, challenge.get("current_streak", 0)) - 1
    
    # Recalculate derived stats
    total = challenge["wins"] + challenge["losses"]
    challenge["total_trades"] = total
    if total > 0:
        challenge["win_rate"] = round(challenge["wins"] / total, 4)


def record_trade_result(pnl_amount: float, signal_id: Optional[str] = None) -> None:
    """
    Record a completed trade result and update challenge stats.
    """
    data = load_data()
    _update_challenge_with_pnl(data, pnl_amount)
    
    # Update signal status if linked
    if signal_id:
        for signal in data.get("signals", []):
            if signal.get("signal_id") == signal_id:
                signal["status"] = "closed"
                signal["final_pnl"] = pnl_amount
                break
    
    save_data(data)
    logger.info(f"Recorded trade result: PnL={pnl_amount}, signal={signal_id}")


def reset_challenge() -> None:
    """Reset challenge to starting state."""
    data = load_data()
    start_balance = float(os.getenv("CHALLENGE_START", "100.0"))
    
    data["challenge"] = {
        "balance": start_balance,
        "start_balance": start_balance,
        "wins": 0,
        "losses": 0,
        "total_trades": 0,
        "win_rate": 0.0,
        "current_streak": 0,
        "max_drawdown": 0.0
    }
    
    # Archive old signals
    old_signals = data.get("signals", [])
    if old_signals:
        archive_file = BACKUP_DIR / f"signals_archive_{datetime.utcnow():%Y%m%d_%H%M%S}.json"
        with open(archive_file, "w") as f:
            json.dump(old_signals, f, indent=2)
        logger.info(f"Archived {len(old_signals)} signals to {archive_file}")
    
    data["signals"] = []
    save_data(data, backup=False)  # No backup for reset
    logger.info("Challenge reset complete")


# ----- Utility -----
def get_storage_stats() -> Dict[str, Any]:
    """Get storage statistics and health info."""
    ensure_storage()
    
    data = load_data()
    
    stats = {
        "storage_dir": str(STORAGE_DIR),
        "data_file_size": DATA_FILE.stat().st_size if DATA_FILE.exists() else 0,
        "total_signals": len(data.get("signals", [])),
        "total_pnl_records": len(data.get("pnl", [])),
        "challenge_balance": data.get("challenge", {}).get("balance", 0),
        "last_updated": data.get("stats", {}).get("last_updated"),
        "backups_available": len(list(BACKUP_DIR.glob("data_*.json"))),
        "version": data.get("version", 1)
    }
    
    return stats


def export_data(filepath: Optional[str] = None) -> str:
    """Export all data to JSON file."""
    if filepath is None:
        filepath = STORAGE_DIR / f"export_{datetime.utcnow():%Y%m%d_%H%M%S}.json"
    else:
        filepath = Path(filepath)
    
    data = load_data()
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"Data exported to {filepath}")
    return str(filepath)


# ----- Exports -----
__all__ = [
    "ensure_storage",
    "load_data",
    "save_data",
    "record_signal",
    "get_signal_history",
    "update_signal_status",
        "link_pnl_to_signal",
    "record_pnl_screenshot",
    "get_pnl_history",
    "get_challenge_stats",
    "update_challenge_balance",
    "record_trade_result",
    "reset_challenge",
    "get_storage_stats",
    "export_data",
    "SignalRecord",
    "PnLRecord",
    "ChallengeStats"
]


# ----- Self-Test -----
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("=" * 50)
    print("Storage Module Self-Test")
    print("=" * 50)
    
    # Test initialization
    print("\n1. Testing initialization:")
    ensure_storage()
    print(f"   Storage dir: {STORAGE_DIR}")
    print(f"   Data file: {DATA_FILE}")
    print(f"   PnL dir: {PNL_DIR}")
    
    # Test data operations
    print("\n2. Testing data operations:")
    data = load_data()
    print(f"   Loaded version: {data.get('version')}")
    print(f"   Signals count: {len(data.get('signals', []))}")
    
    # Test signal recording
    print("\n3. Testing signal recording:")
    test_signal = {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "signal": "LONG",
        "entry": 50000.0,
        "sl": 49000.0,
        "tp1": 52000.0,
        "confidence": 0.85,
        "user_id": 12345,
        "auto": False
    }
    sid = record_signal(test_signal)
    print(f"   Recorded signal ID: {sid}")
    
    # Test challenge stats
    print("\n4. Testing challenge stats:")
    stats = get_challenge_stats()
    print(f"   Balance: {stats.balance}")
    print(f"   Win rate: {stats.win_rate}")
    
    # Test storage stats
    print("\n5. Testing storage stats:")
    health = get_storage_stats()
    print(f"   File size: {health['data_file_size']} bytes")
    print(f"   Total signals: {health['total_signals']}")
    
    print("\n" + "=" * 50)
    print("Self-test complete")
    print("=" * 50)
    
    
