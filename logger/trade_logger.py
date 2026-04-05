import logging
import os
from datetime import datetime, timezone

LOG_DIR = "logs"
TRADE_LOG_FILE = os.path.join(LOG_DIR, "trades.log")


def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    if not root.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
        )

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)

    return root


def get_logger(name="bot"):
    setup_logging()
    return logging.getLogger(name)


class TradeLogger:
    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        self.filepath = TRADE_LOG_FILE

    def _write(self, tag, **data):
        line = f"{datetime.now(timezone.utc).isoformat()} | {tag} | {data}\n"
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(line)

    def log_system(self, msg):
        self._write("SYSTEM", msg=msg)

    def log_scan_start(self, symbols):
        self._write("SCAN_START", symbols=symbols)

    def log_scan_end(self, symbols=None, elapsed=None, signals=None, executed=None):
        self._write(
            "SCAN_END",
            symbols=symbols,
            elapsed=elapsed,
            signals=signals,
            executed=executed,
        )

    def log_trade(self, **kwargs):
        self._write("TRADE", **kwargs)

    def log_close(self, **kwargs):
        self._write("CLOSE", **kwargs)

    def __getattr__(self, name):
        def fallback(*args, **kwargs):
            self._write(name.upper(), args=args, kwargs=kwargs)
        return fallback
