"""Where things live. Nothing here spells out an absolute path.

    MTAH_DATA_DIR       folder holding tactical/                 default: <repo>/data
    MTAH_CACHE_DIR      parsed arrays of the open matches        default: $MTAH_DATA_DIR/cache
    MTAH_OFFICIAL_REPO  the organisers' repository               default: $MTAH_DATA_DIR/more-than-a-hack-2027

The match data is never committed; neither is anything under the cache.
"""

import os
from pathlib import Path

GAME = "BAR-ATH"
FPS = 25

SUFFIXES = {
    "video": ".mp4",
    "atd_meta": "_atd.xml",
    "atd_raw": "_atd.txt",
    "homography": "_homography.csv",
    "smart_tagging": "_smart_tagging.xml",
}

REPO_DIR = Path(__file__).resolve().parent.parent


def data_dir():
    return Path(os.environ.get("MTAH_DATA_DIR") or REPO_DIR / "data")


def cache_dir():
    path = Path(os.environ.get("MTAH_CACHE_DIR") or data_dir() / "cache")
    path.mkdir(parents=True, exist_ok=True)
    return path


def official_repo():
    return Path(os.environ.get("MTAH_OFFICIAL_REPO") or data_dir() / "more-than-a-hack-2027")


def asset(name, source="tactical"):
    """Path of one provided file; not checked for existence."""
    return data_dir() / source / "{}_{}{}".format(GAME, source, SUFFIXES[name])


def require(path):
    """The path if the file exists, else a readable error saying where it was expected."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            "{} not found.\nSet MTAH_DATA_DIR to the folder that contains tactical/ "
            "(currently {}).".format(path, data_dir()))
    return path
