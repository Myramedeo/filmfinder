"""
download.py
-----------
Downloads and extracts the MovieLens 100K dataset.

Usage:
    python -m data.download                  # saves to data/raw/
    python -m data.download --dest my/path
"""

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DEFAULT_DEST = Path(__file__).parent / "raw"


def download_movielens(dest: Path = DEFAULT_DEST, force: bool = False) -> Path:
    """Download and extract MovieLens 100K. Returns path to extracted folder."""
    dest = Path(dest)
    extracted = dest / "ml-100k"

    if extracted.exists() and not force:
        print(f"[skip] Already exists at {extracted}. Pass --force to re-download.")
        return extracted

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MovieLens 100K from {MOVIELENS_URL} …")

    req = urllib.request.Request(
        MOVIELENS_URL,
        headers={"User-Agent": "Mozilla/5.0 (compatible; movie-recommender-project)"},
    )
    with urllib.request.urlopen(req) as resp:
        data = resp.read()

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(dest)

    print(f"Extracted to {extracted}")
    return extracted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default=str(DEFAULT_DEST))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download_movielens(Path(args.dest), force=args.force)