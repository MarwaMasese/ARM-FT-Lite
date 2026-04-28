"""
Image URL loader utility.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# A small set of Creative-Commons / public-domain test images
_FALLBACK_URLS: List[str] = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Dog_Breeds.jpg/320px-Dog_Breeds.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Good_Food_Display_-_NCI_Visuals_Online.jpg/320px-Good_Food_Display_-_NCI_Visuals_Online.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg",
]


def load_image_urls(path: str | Path, limit: int = 100) -> List[str]:
    """
    Load image URLs from a plain text file (one URL per line).

    Falls back to a built-in list of public-domain test images if the file
    does not exist, repeating entries as needed to reach *limit*.

    Parameters
    ----------
    path:
        Path to a text file containing image URLs.
    limit:
        Maximum number of URLs to return.

    Returns
    -------
    list of str
    """
    file_path = Path(path)
    urls: List[str] = []

    if file_path.is_file():
        with file_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    urls.append(line)
        logger.info("Loaded %d URLs from %s", len(urls), file_path)
    else:
        logger.warning(
            "URL file not found (%s).  Using %d built-in fallback URLs, repeated to %d.",
            file_path,
            len(_FALLBACK_URLS),
            limit,
        )
        while len(urls) < limit:
            urls.extend(_FALLBACK_URLS)

    return urls[:limit]
