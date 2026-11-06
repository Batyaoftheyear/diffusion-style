"""Ограниченные повторы HTTP и продолжение чтения каталога без сети."""
from contextlib import redirect_stdout
from email.message import Message
from email.utils import formatdate
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from src.config import load_config
from src.data import DataDownloadError, fetch, retry_delay, viewer_page
from src.io import write_json


class Response(io.BytesIO):
    def __init__(self, body):
        super().__init__(body)
        self.headers = {}


def http_error(code, retry_after=None):
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return HTTPError("https://example.test/rows", code, "temporary test error", headers, io.BytesIO())


class DownloadTests(unittest.TestCase):
    def test_service_unavailable_then_success(self):
        output = io.StringIO()
        with patch("src.data.urlopen", side_effect=[http_error(503), http_error(502), Response(b"recovered")]) as request, patch("src.data.time.sleep") as sleep, redirect_stdout(output):
            body, _ = fetch("https://example.test/image?Signature=do-not-log-this")
        self.assertEqual(body, b"recovered")
        self.assertEqual(request.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 4])
        self.assertIn("HTTP 503", output.getvalue())
        self.assertNotIn("do-not-log-this", output.getvalue())

    def test_retry_after_and_http_date(self):
        with patch("src.data.urlopen", side_effect=[http_error(503, "7"), Response(b"ok")]), patch("src.data.time.sleep") as sleep, redirect_stdout(io.StringIO()):
            self.assertEqual(fetch("https://example.test/rows")[0], b"ok")
        sleep.assert_called_once_with(7)
        with patch("src.data.time.time", return_value=1_700_000_000):
            self.assertEqual(retry_delay(0, formatdate(1_700_000_011, usegmt=True)), 11)
        self.assertEqual(retry_delay(0, "invalid"), 2)
        self.assertEqual(retry_delay(0, "999"), 60)

    def test_permanent_error_is_not_retried(self):
        with patch("src.data.urlopen", side_effect=http_error(404)) as request, patch("src.data.time.sleep") as sleep:
            with self.assertRaises(HTTPError):
                fetch("https://example.test/missing")
        self.assertEqual(request.call_count, 1)
        sleep.assert_not_called()

    def test_exhausted_retries_are_explicit(self):
        with patch("src.data.urlopen", side_effect=[http_error(503) for _ in range(6)]) as request, patch("src.data.time.sleep") as sleep, redirect_stdout(io.StringIO()):
            with self.assertRaises(DataDownloadError) as caught:
                fetch("https://example.test/rows")
        self.assertEqual(request.call_count, 6)
        self.assertEqual(sleep.call_count, 5)
        self.assertIn("preserved", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, HTTPError)

    def test_pinned_metadata_cache_survives_image_url_expiration(self):
        config = load_config()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache/wikiart_0_100.json"
            payload = {"rows": [], "features": []}
            write_json(cache, {"revision": config["art"]["revision"], "payload": payload})
            os.utime(cache, (1, 1))
            with patch("src.data.fetch", side_effect=AssertionError("Metadata must resume from the cached pinned page")):
                self.assertEqual(viewer_page(config, root, 0, fresh_urls=False), payload)
            with patch("src.data.fetch", return_value=(json.dumps(payload).encode(), {"x-revision": config["art"]["revision"]})) as request:
                self.assertEqual(viewer_page(config, root, 0, fresh_urls=True), payload)
                request.assert_called_once()

    def test_wrong_cached_revision_is_rejected(self):
        config = load_config()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "cache/wikiart_0_100.json", {"revision": "0" * 40, "payload": {"rows": []}})
            with patch("src.data.fetch", side_effect=AssertionError("No silent source replacement")):
                with self.assertRaisesRegex(ValueError, "revision"):
                    viewer_page(config, root, 0, fresh_urls=False)


if __name__ == "__main__":
    unittest.main()
