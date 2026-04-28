"""
Unit tests for the image downloader.
"""

from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from pipeline.downloader import download_image


def _make_response(image: Image.Image, status_code: int = 200):
    buf = BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)

    resp = MagicMock()
    resp.status_code = status_code
    resp.content = buf.read()
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        import requests
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    return resp


class TestDownloadImage:
    def test_returns_pil_image(self):
        fake_image = Image.new("RGB", (100, 100), color=(255, 0, 0))
        mock_resp = _make_response(fake_image)

        with patch("pipeline.downloader.requests.get", return_value=mock_resp):
            result = download_image("http://example.com/img.jpg")

        assert isinstance(result, Image.Image)

    def test_http_error_raises(self):
        import requests as req_lib

        mock_resp = _make_response(Image.new("RGB", (10, 10)), status_code=404)

        with patch("pipeline.downloader.requests.get", return_value=mock_resp):
            with pytest.raises(req_lib.HTTPError):
                download_image("http://example.com/missing.jpg")

    def test_invalid_body_raises_value_error(self):
        resp = MagicMock()
        resp.content = b"not-an-image"
        resp.raise_for_status = MagicMock()

        with patch("pipeline.downloader.requests.get", return_value=resp):
            with pytest.raises(ValueError, match="Could not decode image"):
                download_image("http://example.com/corrupt.jpg")
