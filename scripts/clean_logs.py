"""
Clean log files: remove tqdm progress bars and ANSI escape codes.

Usage:
    python scripts/clean_logs.py                             # clean all in results/logs/
    python scripts/clean_logs.py --file path/to/file.log    # single file
    python scripts/clean_logs.py --inplace                   # overwrite originals (default: *.clean.log)
"""
import argparse
import re
import sys
from pathlib import Path

LOG_DIR = Path("results/logs")

ANSI_RE = re.compile(rb"\x1b\[[0-9;]*[mABCDEFGHJKSTfhilmnprsu]")
TQDM_RE = re.compile(r"\d+%\|")


def clean_file(path: Path, inplace: bool) -> None:
    raw = ANSI_RE.sub(b"", path.read_bytes())
    text = raw.decode("utf-8", errors="replace")

    out = []
    for line in text.split("\n"):
        final = line.split("\r")[-1].strip()
        if not final or TQDM_RE.search(final):
            continue
        out.append(final)

    out_path = path if inplace else path.with_suffix(".clean" + path.suffix)
    out_path.write_bytes(("\n".join(out) + "\n").encode("utf-8"))
    print(f"{path.name}: {text.count(chr(10))} → {len(out)} lines → {out_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove progress bars from log files")
    parser.add_argument("--file", default=None, help="Single log file to clean")
    parser.add_argument("--dir", default=str(LOG_DIR), help="Log directory (default: results/logs/)")
    parser.add_argument("--inplace", action="store_true", help="Overwrite originals")
    args = parser.parse_args()

    if args.file:
        targets = [Path(args.file)]
    else:
        log_dir = Path(args.dir)
        if not log_dir.exists():
            print(f"[ERROR] {log_dir} not found.")
            sys.exit(1)
        targets = sorted(log_dir.glob("*.log"))
        if not targets:
            print(f"No .log files in {log_dir}")
            sys.exit(0)

    for path in targets:
        clean_file(path, inplace=args.inplace)
