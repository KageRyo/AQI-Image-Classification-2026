"""Download and extract the public Kaggle competition dataset."""

from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path

import kagglehub

COMPETITION = "2026-dl-final-exam-india-nepal-aqi-classification"
REQUIRED_FILES = [
    "train_data.csv",
    "val_data.csv",
    "test_data.csv",
    "sample_submission.csv",
]


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition("=")
        if separator:
            os.environ.setdefault(key.strip(), value.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--force", action="store_true", help="Overwrite existing extracted files.")
    return parser.parse_args()


def copy_tree(source: Path, destination: Path, force: bool) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        relative_path = path.relative_to(source)
        output_path = destination / relative_path
        if path.is_dir():
            output_path.mkdir(parents=True, exist_ok=True)
        elif force or not output_path.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, output_path)


def extract_download(download_path: Path, output_dir: Path, force: bool) -> None:
    if download_path.is_file() and zipfile.is_zipfile(download_path):
        with zipfile.ZipFile(download_path) as archive:
            if force:
                archive.extractall(output_dir)
            else:
                for member in archive.infolist():
                    destination = output_dir / member.filename
                    if not destination.exists():
                        archive.extract(member, output_dir)
    elif download_path.is_dir():
        copy_tree(download_path, output_dir, force)
    else:
        raise RuntimeError(f"Unsupported Kaggle download path: {download_path}")


def find_one(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {name} below {root}, found {len(matches)}.")
    return matches[0]


def validate_dataset(output_dir: Path) -> None:
    for filename in REQUIRED_FILES:
        find_one(output_dir, filename)
    image_dirs = [path for path in output_dir.rglob("images") if path.is_dir()]
    if len(image_dirs) != 1:
        raise RuntimeError(f"Expected one images/ directory below {output_dir}, found {len(image_dirs)}.")
    image_count = sum(
        path.suffix.lower() in {".jpg", ".jpeg", ".png"} for path in image_dirs[0].rglob("*")
    )
    if image_count == 0:
        raise RuntimeError(f"No images found below {image_dirs[0]}.")
    print(f"Dataset ready: {output_dir.resolve()}")
    print(f"Images found: {image_count}")


def main() -> None:
    args = parse_args()
    load_dotenv()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    download_path = Path(kagglehub.competition_download(COMPETITION))
    print(f"Kaggle download path: {download_path}")
    extract_download(download_path, args.output_dir, args.force)
    validate_dataset(args.output_dir)


if __name__ == "__main__":
    main()
