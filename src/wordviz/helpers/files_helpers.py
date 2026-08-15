import gzip
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import logging

logger = logging.getLogger(__name__)

VALID_EXT = [".bin", ".txt", ".vec"]


def get_cache_dir() -> Path:
    cache_dir = Path.home() / ".wordviz_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_file(url: str, filename: str) -> Path:
    """downloads file from url into the cache directory"""
    download_path = get_cache_dir() / filename
    if not download_path.exists():
        logger.info(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, download_path)
    else:
        logger.info(f"{filename} already exists in cache.")
    return download_path


def validate_file(path: str) -> bool:
    """checks if path argument leads to a valid file name and returns if it is binary"""
    if path is None:
        raise ValueError("File path is required")
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("The file path must be a string")

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Invalid file path {path}: the file does not exist")

    ext = path.suffix.lower()
    if ext in (".gz", ".zip"):
        raise ValueError(
            "Compressed files are not supported. Please extract the file first."
        )
    if ext not in VALID_EXT:
        raise ValueError(
            f"Invalid file extension {ext}. Valid extensions are: {','.join(VALID_EXT)}"
        )

    return ext == ".bin"


def validate_zip(zip_path: Path, member: str) -> None:
    """checks that the downloaded file is a valid zip archive and contains the expected file"""
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Downloaded file is not a valid zip archive: {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        if member not in z.namelist():
            raise ValueError(f"File '{member}' not found in zip archive {zip_path}")


def validate_gzip(gz_path: Path) -> None:
    """checks that the downloaded file is a valid gzip archive"""
    try:
        with gzip.open(gz_path, "rb") as f:
            f.read(1)
    except OSError as e:
        raise ValueError(
            f"Downloaded file is not a valid gzip archive: {gz_path}"
        ) from e


def extract_archive(archive_path: Path, member: str, dest_dir: Path) -> None:
    """extracts member from a zip/gzip archive, or copies the file directly, into dest_dir"""
    if archive_path.suffix.lower() == ".zip":
        validate_zip(archive_path, member)
        logger.info(f"Extracting {member}...")
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extract(member, path=dest_dir)
    elif archive_path.suffix.lower() == ".gz":
        validate_gzip(archive_path)
        logger.info(f"Decompressing {member}...")
        with gzip.open(archive_path, "rb") as src, open(dest_dir / member, "wb") as dst:
            shutil.copyfileobj(src, dst)
    else:
        logger.info(f"Copying {member}...")
        shutil.copyfile(archive_path, dest_dir / member)


def export_embedding(source_path, dest_folder):
    """saves locally pretrained embeddings file"""
    os.makedirs(dest_folder, exist_ok=True)
    filename = os.path.basename(source_path)
    dest_path = os.path.join(dest_folder, filename)
    shutil.copy(source_path, dest_path)
    logger.info(f"File saved in {dest_path}.")
