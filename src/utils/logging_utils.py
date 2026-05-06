"""Tee stdout+stderr to results/logs/<script>_<timestamp>.log."""
import atexit
import datetime
import sys
from pathlib import Path

LOG_DIR = Path("results/logs")


class _Tee:
    def __init__(self, stream, logfile):
        self._stream = stream
        self._log = logfile

    def write(self, data):
        self._stream.write(data)
        self._log.write(data)
        return len(data)

    def flush(self):
        self._stream.flush()
        self._log.flush()

    def isatty(self):
        return self._stream.isatty()

    def fileno(self):
        return self._stream.fileno()

    @property
    def encoding(self):
        return self._stream.encoding

    @property
    def errors(self):
        return self._stream.errors


def setup_logging(script_path: str) -> Path:
    """Tee stdout+stderr to results/logs/<stem>_<timestamp>.log. Returns log path."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(script_path).stem
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{stem}_{ts}.log"

    logfile = open(log_path, "w", encoding="utf-8", buffering=1)

    header = (
        f"=== {stem} | {datetime.datetime.now().isoformat(timespec='seconds')} ===\n"
        f"argv: {' '.join(sys.argv)}\n"
        f"{'='*60}\n"
    )
    logfile.write(header)

    sys.stdout = _Tee(sys.__stdout__, logfile)
    sys.stderr = _Tee(sys.__stderr__, logfile)

    def _finish():
        end = datetime.datetime.now().isoformat(timespec="seconds")
        try:
            sys.stdout.write(f"\n=== done {end} | log → {log_path} ===\n")
        except Exception:
            pass
        logfile.flush()
        logfile.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    atexit.register(_finish)
    print(f"Logging → {log_path}")
    return log_path
