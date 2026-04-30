"""Dataset download helpers. Run via scripts/00_download.sh or manually."""
import os
import subprocess
import zipfile
import tarfile
from pathlib import Path

RAW = Path("data/raw")


def _wget(url: str, dest: Path, fname: str | None = None) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / (fname or url.split("/")[-1])
    if out.exists():
        print(f"[skip] {out} already exists")
        return out
    subprocess.run(["wget", "-q", "--show-progress", "-O", str(out), url], check=True)
    return out


def _unzip(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    elif archive.suffix in (".tar", ".gz", ".bz2"):
        with tarfile.open(archive) as tf:
            tf.extractall(dest)
    else:
        raise ValueError(f"Unknown archive type: {archive.suffix}")


# ── Download instructions ──────────────────────────────────────────────────
# These datasets require form registration or specific licenses.
# Provide manual download URLs or agree to license terms on the dataset pages.

DATASET_URLS = {
    "logodet3k": {
        "info": "https://github.com/Wangjing1551/LogoDet-3K-Dataset",
        "notes": (
            "LogoDet-3K: Download from the GitHub page above. "
            "Place extracted contents in data/raw/LogoDet-3K/ "
            "(giữ nguyên cấu trúc thư mục: data/raw/LogoDet-3K/{category}/{ClassName}/)."
        ),
    },
}


def print_download_instructions() -> None:
    print("=" * 70)
    print("MANUAL DOWNLOAD REQUIRED")
    print("=" * 70)
    for name, info in DATASET_URLS.items():
        dest = RAW / name
        exists = dest.exists() and any(dest.iterdir())
        status = "[OK]" if exists else "[MISSING]"
        print(f"\n{status} {name.upper()}")
        print(f"  URL : {info['info']}")
        print(f"  Note: {info['notes']}")
    print("=" * 70)


def check_datasets() -> dict[str, bool]:
    results = {}
    for name in DATASET_URLS:
        dest = RAW / name
        ok = dest.exists() and any(dest.iterdir())
        results[name] = ok
        symbol = "✓" if ok else "✗"
        print(f"  {symbol} {name}: {dest}")
    return results


if __name__ == "__main__":
    print_download_instructions()
    print("\nChecking existing data:")
    status = check_datasets()
    missing = [k for k, v in status.items() if not v]
    if missing:
        print(f"\nMissing: {missing}. Download manually then re-run.")
    else:
        print("\nLogoDet-3K dataset present.")
