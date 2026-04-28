"""
Image downloader module.
Downloads images from URLs into PIL Image objects.
"""

from __future__ import annotations

import logging
from io import BytesIO

import requests
from PIL import Image

logger = logging.getLogger(__name__)

# Timeout for HTTP requests (connect, read)
_REQUEST_TIMEOUT = (5, 30)


def download_image(url: str) -> Image.Image:
    """
    Download an image from *url* and return it as a PIL Image.

    Parameters
    ----------
    url:
        Public URL pointing to a JPEG / PNG image.

    Returns
    -------
    PIL.Image.Image
        The downloaded image, mode unchanged.

    Raises
    ------
    requests.HTTPError
        If the server returns a non-2xx status code.
    ValueError
        If the response body cannot be decoded as an image.
    """
    logger.debug("Downloading image from %s", url)

    response = requests.get(url, timeout=_REQUEST_TIMEOUT)
    response.raise_for_status()

    try:
        image = Image.open(BytesIO(response.content))
        image.load()  # Force decoding while BytesIO is still alive
    except Exception as exc:  # PIL raises a broad set of exceptions
        raise ValueError(f"Could not decode image from {url}: {exc}") from exc

    logger.debug("Downloaded image size=%s mode=%s", image.size, image.mode)
    return image
