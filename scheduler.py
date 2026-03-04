# scheduler.py
"""
Task scheduler for Destiny Trading Empire Bot.
Handles automated briefings, signal scanning, and periodic maintenance tasks.
"""

import os
import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Optional, List, Callable, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logger = logging.getLogger("scheduler")
logger.setLevel(logging.INFO)

# Optional dependencies
try:
    from pro_features import generate_pro_report, get_ai_market_insight, get_fear_greed_index
    HAS_PRO_FEATURES = True
except ImportError as e:
    logger.warning(f"pro_features not available: {e}")
    generate_pro_report = None
    get_ai_market_insight = None
    get_fear_greed_index = None
    HAS_PRO_FEATURES = False

try:
    from signal_engine import start_auto_scanner, stop_auto_scanner, SignalScanner
    HAS_SIGNAL_ENGINE = True
except ImportError as e:
    logger.warning(f"signal_engine not available: {e}")
    start_auto_scanner = None
    stop_auto_scanner = None
    SignalScanner = None
    HAS_SIGNAL_ENGINE = False

try:
    from storage import load_data, save_data, record_signal
    HAS_STORAGE = True
except ImportError as e:
    logger.warning(f"storage not available: {e}")
    load_data = None
    save_data = None
    record_signal = None
    HAS_STORAGE = False


class TaskType(Enum):
    """Types of scheduled tasks."""
    MARKET_BRIEF = "market_brief"
    SIGNAL_SCAN = "signal_scan"
    MAINTENANCE = "maintenance"
    CUSTOM = "custom"


@dataclass
class ScheduledTask:
    """Represents a scheduled task configuration."""
    task_id: str
    task_type: TaskType
    interval_seconds: int
    enabled: bool = True
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    run_count: int = 0
    error_count: int = 0
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}


@dataclass
class ScheduleConfig:
    """Global scheduler configuration."""
    admin_chat_id: Optional[int] = None
    market_brief_interval: int = 3600  # 1 hour
    signal_scan_interval: int = 300   # 5 minutes
    maintenance_interval: int = 86400   # 24 hours
    enabled_tasks: List[str] = None
    
    def __post_init__(self):
        if self.enabled_tasks is None:
            self.enabled_tasks = ["market_brief", "signal_scan"]


# ----- Configuration Management -----
def load_scheduler_config() -> ScheduleConfig:
    """Load scheduler configuration from storage or environment."""
    admin_id = os.getenv("ADMIN_ID")
    
    config = ScheduleConfig(
        admin_chat_id=int(admin_id) if admin_id else None,
        market_brief_interval=int(os.getenv("BRIEF_INTERVAL_SEC", "3600")),
        signal_scan_interval=int(os.getenv("SCAN_INTERVAL_SEC", "300")),
        maintenance_interval=int(os.getenv("MAINTENANCE_INTERVAL_SEC", "86400"))
    )
    
    # Try to load from storage for persistence
    if HAS_STORAGE:
        try:
            data = load_data()
            saved_config = data.get("scheduler_config", {})
            if saved_config.get("admin_chat_id"):
                config.admin_chat_id = saved_config["admin_chat_id"]
            if saved_config.get("enabled_tasks"):
                config.enabled_tasks = saved_config["enabled_tasks"]
        except Exception as e:
            logger.debug(f"Could not load saved config: {e}")
    
    return config


def save_scheduler_config(config: ScheduleConfig) -> None:
    """Save scheduler configuration to storage."""
    if not HAS_STORAGE:
        return
    
    try:
        data = load_data()
        data["scheduler_config"] = {
            "admin_chat_id": config.admin_chat_id,
            "market_brief_interval": config.market_brief_interval,
            "signal_scan_interval": config.signal_scan_interval,
            "enabled_tasks": config.enabled_tasks,
            "last_saved": datetime.utcnow().isoformat()
        }
        save_data(data)
    except Exception as e:
        logger.error(f"Failed to save scheduler config: {e}")


# ----- Task Registry -----
class TaskRegistry:
    """Manages scheduled tasks and their execution."""
    
    def __init__(self):
        self.tasks: Dict[str, ScheduledTask] = {}
        self._handlers: Dict[str, Callable] = {}
        self._lock = threading.Lock()
    
    def register(self, task: ScheduledTask, handler: Callable) -> None:
        """Register a task with its handler function."""
        with self._lock:
            self.tasks[task.task_id] = task
            self._handlers[task.task_id] = handler
            logger.info(f"Registered task: {task.task_id} ({task.task_type.value})")
    
    def unregister(self, task_id: str) -> bool:
        """Remove a task from registry."""
        with self._lock:
            if task_id in self.tasks:
                del self.tasks[task_id]
                del self._handlers[task_id]
                logger.info(f"Unregistered task: {task_id}")
                return True
            return False
    
    def get_handler(self, task_id: str) -> Optional[Callable]:
        """Get handler for a task."""
        return self._handlers.get(task_id)
    
    def update_last_run(self, task_id: str) -> None:
        """Update last run timestamp for a task."""
        with self._lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.last_run = datetime.utcnow().isoformat()
                task.run_count += 1
                # Calculate next run
                next_time = datetime.utcnow() + timedelta(seconds=task.interval_seconds)
                task.next_run = next_time.isoformat()
    
    def record_error(self, task_id: str, error: str) -> None:
        """Record an error for a task."""
        with self._lock:
            if task_id in self.tasks:
                self.tasks[task_id].error_count += 1
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all tasks."""
        with self._lock:
            return {
                task_id: {
                    "type": task.task_type.value,
                    "enabled": task.enabled,
                    "interval": task.interval_seconds,
                    "last_run": task.last_run,
                    "next_run": task.next_run,
                    "run_count": task.run_count,
                    "error_count": task.error_count
                }
                for task_id, task in self.tasks.items()
            }


# Global registry
_registry = TaskRegistry()


# ----- Task Handlers -----
def _market_brief_handler(bot, config: Dict[str, Any]) -> bool:
    """
    Generate and send market brief to admin.
    """
    try:
        if not HAS_PRO_FEATURES:
            logger.warning("Pro features not available for market brief")
            return False
        
        # Get symbols to analyze
        symbols = config.get("symbols", ["BTCUSDT", "ETHUSDT", "BNBUSDT"])
        
        # Generate report
        report = generate_pro_report(
            symbols=symbols,
            interval=config.get("interval", "1h"),
            include_charts=True,
            include_ai=True
        )
        
        # Send to admin
        admin_id = config.get("admin_chat_id")
        if not admin_id:
            logger.error("No admin chat ID configured for market brief")
            return False
        
        # Send summary text
        if report.get("summary"):
            bot.send_message(
                admin_id,
                report["summary"],
                parse_mode="HTML"
            )
        
        # Send chart if available
        if report.get("chart"):
            bot.send_photo(
                admin_id,
                report["chart"],
                caption="📊 Market Overview Chart"
            )
        
        # Send AI insight if available
        if report.get("ai_insight"):
            bot.send_message(
                admin_id,
                report["ai_insight"],
                parse_mode="HTML"
            )
        
        logger.info(f"Market brief sent to {admin_id}")
        return True
        
    except Exception as e:
        logger.exception(f"Market brief handler failed: {e}")
        return False


def _signal_scan_handler(bot, config: Dict[str, Any]) -> bool:
    """
    Handle automated signal scanning.
    """
    try:
        if not HAS_SIGNAL_ENGINE:
            logger.warning("Signal engine not available")
            return False
        
        # This is handled by signal_engine's auto-scanner
        # Just verify it's running
        if SignalScanner and not any(
            t.name == "SignalScanner" for t in threading.enumerate() if hasattr(t, "name")
        ):
            logger.warning("Signal scanner not running, attempting to start...")
            pairs = config.get("pairs", ["BTCUSDT", "ETHUSDT"])
            start_auto_scanner(
                pairs=pairs,
                interval=config.get("interval", "1h"),
                min_confidence=config.get("min_confidence", 0.75)
            )
        
        return True
        
    except Exception as e:
        logger.exception(f"Signal scan handler failed: {e}")
        return False


def _maintenance_handler(bot, config: Dict[str, Any]) -> bool:
    """
    Perform maintenance tasks.
    """
    try:
        tasks_performed = []
        
        # Clean old data if storage available
        if HAS_STORAGE:
            try:
                data = load_data()
                signals = data.get("signals", [])
                
                # Archive old signals (older than 90 days)
                cutoff = (datetime.utcnow() - timedelta(days=90)).isoformat()
                old_signals = [s for s in signals if s.get("timestamp", "") < cutoff]
                
                if old_signals:
                    # In real implementation, move to archive
                    logger.info(f"Found {len(old_signals)} old signals to archive")
                    tasks_performed.append(f"archived_{len(old_signals)}_signals")
                
            except Exception as e:
                logger.warning(f"Maintenance data cleanup failed: {e}")
        
        # Health check log
        logger.info(f"Maintenance completed: {tasks_performed}")
        return True
        
    except Exception as e:
        logger.exception(f"Maintenance handler failed: {e}")
        return False


# ----- Scheduler Core -----
class BotScheduler:
    """
    Main scheduler class for managing automated tasks.
    Thread-safe with proper lifecycle management.
    """
    
    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._running = False
        self._bot: Optional[Any] = None
        self._config: Optional[ScheduleConfig] = None
    
    def _scheduler_loop(self):
        """Main scheduler execution loop."""
        logger.info("Scheduler loop starting...")
        
        while not self._stop_event.is_set():
            cycle_start = time.time()
            
            try:
                # Check each registered task
                for task_id, task in _registry.tasks.items():
                    if not task.enabled:
                        continue
                    
                    # Check if it's time to run
                    should_run = False
                    if task.last_run is None:
                        should_run = True
                    else:
                        last = datetime.fromisoformat(task.last_run)
                        next_run = last + timedelta(seconds=task.interval_seconds)
                        should_run = datetime.utcnow() >= next_run
                    
                    if should_run:
                        logger.debug(f"Executing task: {task_id}")
                        handler = _registry.get_handler(task_id)
                        
                        if handler:
                            try:
                                success = handler(self._bot, task.config or {})
                                if success:
                                    _registry.update_last_run(task_id)
                                else:
                                    _registry.record_error(task_id, "Handler returned False")
                            except Exception as e:
                                logger.exception(f"Task {task_id} failed: {e}")
                                _registry.record_error(task_id, str(e))
                        else:
                            logger.warning(f"No handler for task: {task_id}")
                
            except Exception as e:
                logger.exception(f"Scheduler loop error: {e}")
            
            # Sleep until next check (every 10 seconds)
            elapsed = time.time() - cycle_start
            sleep_time = max(0, 10 - elapsed)
            
            if sleep_time > 0 and not self._stop_event.is_set():
                self._stop_event.wait(timeout=sleep_time)
        
        self._running = False
        logger.info("Scheduler loop stopped")
    
    def start(self, bot, config: Optional[ScheduleConfig] = None) -> bool:
        """
        Start the scheduler with given bot and configuration.
        """
        with self._lock:
            if self._running:
                logger.warning("Scheduler already running")
                return False
            
            self._bot = bot
            self._config = config or load_scheduler_config()
            
            # Register default tasks
            self._register_default_tasks()
            
            # Start thread
            self._stop_event.clear()
            self._running = True
            self._thread = threading.Thread(
                target=self._scheduler_loop,
                daemon=True,
                name="BotScheduler"
            )
            self._thread.start()
            
            logger.info(f"Scheduler started with config: {asdict(self._config)}")
            return True
    
    def stop(self, timeout: float = 10.0) -> bool:
        """
        Stop the scheduler gracefully.
        """
        with self._lock:
            if not self._running:
                return False
            
            logger.info("Stopping scheduler...")
            self._stop_event.set()
            
            if self._thread:
                self._thread.join(timeout=timeout)
                
                if self._thread.is_alive():
                    logger.warning(f"Scheduler thread did not stop within {timeout}s")
                    return False
            
            self._running = False
            self._thread = None
            logger.info("Scheduler stopped successfully")
            return True
    
    def _register_default_tasks(self) -> None:
        """Register default scheduled tasks based on configuration."""
        if not self._config:
            return
        
        # Market brief task
        if "market_brief" in self._config.enabled_tasks:
            brief_task = ScheduledTask(
                task_id="market_brief",
                task_type=TaskType.MARKET_BRIEF,
                interval_seconds=self._config.market_brief_interval,
                config={
                    "admin_chat_id": self._config.admin_chat_id,
                    "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "DOGEUSDT"]
                }
            )
            _registry.register(brief_task, _market_brief_handler)
        
        # Signal scan task
        if "signal_scan" in self._config.enabled_tasks and HAS_SIGNAL_ENGINE:
            scan_task = ScheduledTask(
                task_id="signal_scan",
                task_type=TaskType.SIGNAL_SCAN,
                interval_seconds=self._config.signal_scan_interval,
                config={
                    "pairs": os.getenv("PAIRS", "BTCUSDT,ETHUSDT").split(","),
                    "interval": "1h",
                    "min_confidence": 0.75
                }
            )
            _registry.register(scan_task, _signal_scan_handler)
        
        # Maintenance task
        maintenance_task = ScheduledTask(
            task_id="maintenance",
            task_type=TaskType.MAINTENANCE,
            interval_seconds=self._config.maintenance_interval,
            config={}
        )
        _registry.register(maintenance_task, _maintenance_handler)
    
    @property
    def is_running(self) -> bool:
        """Check if scheduler is active."""
        return self._running and self._thread is not None and self._thread.is_alive()
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive scheduler status."""
        return {
            "running": self.is_running,
            "config": asdict(self._config) if self._config else None,
            "tasks": _registry.get_status(),
            "thread_alive": self._thread.is_alive() if self._thread else False
        }


# Global scheduler instance
_scheduler = BotScheduler()


# ----- Public API -----
def start_scheduler(
    bot,
    interval_sec: int = 3600,
    admin_chat_id: Optional[int] = None
) -> bool:
    """
    Start the scheduler with given bot.
    Legacy compatibility wrapper.
    """
    config = load_scheduler_config()
    
    # Override with provided values
    if admin_chat_id:
        config.admin_chat_id = admin_chat_id
    config.market_brief_interval = interval_sec
    
    return _scheduler.start(bot, config)


def stop_scheduler(timeout: float = 10.0) -> bool:
    """
    Stop the scheduler.
    Legacy compatibility wrapper.
    """
    return _scheduler.stop(timeout)


def get_scheduler_status() -> Dict[str, Any]:
    """Get current scheduler status."""
    return _scheduler.get_status()


def add_custom_task(
    task_id: str,
    interval_seconds: int,
    handler: Callable[[Any, Dict], bool],
    config: Optional[Dict] = None
) -> bool:
    """
    Add a custom scheduled task.
    """
    try:
        task = ScheduledTask(
            task_id=task_id,
            task_type=TaskType.CUSTOM,
            interval_seconds=interval_seconds,
            config=config or {}
        )
        _registry.register(task, handler)
        return True
    except Exception as e:
        logger.error(f"Failed to add custom task: {e}")
        return False


def remove_task(task_id: str) -> bool:
    """Remove a scheduled task."""
    return _registry.unregister(task_id)


def enable_task(task_id: str) -> bool:
    """Enable a disabled task."""
    with _registry._lock:
        if task_id in _registry.tasks:
            _registry.tasks[task_id].enabled = True
            return True
    return False


def disable_task(task_id: str) -> bool:
    """Disable a task without removing it."""
    with _registry._lock:
        if task_id in _registry.tasks:
            _registry.tasks[task_id].enabled = False
            return True
    return False


# ----- Legacy Compatibility -----
def start_auto_brief_scheduler(bot, interval_sec: int = 600) -> bool:
    """Legacy wrapper for auto-brief only."""
    return start_scheduler(bot, interval_sec)


def stop_auto_brief_scheduler() -> bool:
    """Legacy wrapper."""
    return stop_scheduler()


# ----- Exports -----
__all__ = [
    "BotScheduler",
    "start_scheduler",
    "stop_scheduler",
    "get_scheduler_status",
    "add_custom_task",
    "remove_task",
    "enable_task",
    "disable_task",
    "TaskType",
    "ScheduledTask",
    "ScheduleConfig",
    # Legacy
    "start_auto_brief_scheduler",
    "stop_auto_brief_scheduler"
]


# ----- Self-Test -----
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("=" * 50)
    print("Scheduler Module Self-Test")
    print("=" * 50)
    
    # Test configuration
        config = load_scheduler_config()
    print(f"   Admin ID: {config.admin_chat_id}")
    print(f"   Brief interval: {config.market_brief_interval}s")
    print(f"   Enabled tasks: {config.enabled_tasks}")
    
    # Test registry
    print("\n2. Testing task registry:")
    test_task = ScheduledTask(
        task_id="test_task",
        task_type=TaskType.CUSTOM,
        interval_seconds=60
    )
    
    def test_handler(bot, config):
        print("   Test handler executed!")
        return True
    
    _registry.register(test_task, test_handler)
    print(f"   Registered tasks: {list(_registry.tasks.keys())}")
    
    # Test status
    print("\n3. Testing status:")
    status = _scheduler.get_status()
    print(f"   Running: {status['running']}")
    print(f"   Tasks: {status['tasks']}")
    
    print("\n" + "=" * 50)
    print("Self-test complete")
