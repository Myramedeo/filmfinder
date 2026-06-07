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

MOVIELENS_URLS = {
    "ml-100k": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
    "ml-25m":  "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
}
DEFAULT_DEST = Path(__file__).parent / "raw"


def download_movielens(
    dest: Path = DEFAULT_DEST,
    force: bool = False,
    dataset_name: str = "ml-100k",
) -> Path:
    """Download and extract a MovieLens dataset. Returns path to extracted folder."""
    if dataset_name not in MOVIELENS_URLS:
        raise ValueError(f"Unsupported dataset_name '{dataset_name}'. Choose from: {sorted(MOVIELENS_URLS)}")

    dest = Path(dest)
    extracted = dest / dataset_name
    url = MOVIELENS_URLS[dataset_name]

    if extracted.exists() and not force:
        print(f"[skip] Already exists at {extracted}. Pass --force to re-download.")
        return extracted

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dataset_name.upper()} from {url} …")

    req = urllib.request.Request(
        url,
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
    parser.add_argument("--dataset", default="ml-100k", choices=sorted(MOVIELENS_URLS))
    args = parser.parse_args()
    download_movielens(Path(args.dest), force=args.force, dataset_name=args.dataset)