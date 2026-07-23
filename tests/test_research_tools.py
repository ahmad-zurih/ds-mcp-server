"""
Tests for research_tools.py and screenshot_multi_impl in web_tools.py.
All HTTP calls and external library imports are mocked — no real network access.
"""
from __future__ import annotations

import io
import json
from textwrap import dedent
from unittest.mock import MagicMock, call, patch

import pytest

from ds_mcp_server._tools.research_tools import (
    _extract_video_id,
    arxiv_search_impl,
    github_read_file_impl,
    github_search_impl,
    wikipedia_impl,
    youtube_transcript_impl,
)
from ds_mcp_server._tools.web_tools import screenshot_multi_impl

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ATOM_FEED = dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2301.00001v2</id>
        <title>Attention Is All You Need</title>
        <summary>The dominant sequence transduction models are based on complex recurrent
    or convolutional neural networks.</summary>
        <published>2023-01-01T00:00:00Z</published>
        <author><name>Alice Smith</name></author>
        <author><name>Bob Jones</name></author>
        <link href="https://arxiv.org/pdf/2301.00001v2" rel="related" type="application/pdf"/>
      </entry>
    </feed>
""")

_ATOM_FEED_EMPTY = dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
    </feed>
""")


def _urlopen_ctx(content: str | bytes):
    """Return a mock context-manager that .read() returns `content`."""
    raw = content.encode() if isinstance(content, str) else content
    resp = MagicMock()
    resp.read.return_value = raw
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------


class TestArxivSearch:
    def test_returns_formatted_results(self):
        with patch("urllib.request.urlopen", return_value=_urlopen_ctx(_ATOM_FEED)):
            result = arxiv_search_impl("transformers", max_results=1)
        assert "Attention Is All You Need" in result
        assert "2301.00001" in result
        assert "Alice Smith" in result
        assert "2023-01-01" in result
        assert "arxiv.org/pdf" in result

    def test_empty_results(self):
        with patch("urllib.request.urlopen", return_value=_urlopen_ctx(_ATOM_FEED_EMPTY)):
            result = arxiv_search_impl("xyznonexistent")
        assert "No arXiv results" in result

    def test_network_error(self):
        import urllib.error
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timeout")):
            result = arxiv_search_impl("test")
        assert "arXiv search error" in result

    def test_max_results_capped(self):
        """max_results > 20 should be silently capped to 20."""
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            return _urlopen_ctx(_ATOM_FEED_EMPTY)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            arxiv_search_impl("test", max_results=99)
        assert "max_results=20" in captured["url"]

    def test_multiple_authors_truncated(self):
        many_author_tags = "\n    ".join(
            f"<author><name>Author{i}</name></author>" for i in range(8)
        )
        feed = (
            '<feed xmlns="http://www.w3.org/2005/Atom">'
            "<entry>"
            "<id>http://arxiv.org/abs/9999.00001v1</id>"
            "<title>Multi-Author Paper</title>"
            "<summary>Abstract text.</summary>"
            "<published>2024-01-01T00:00:00Z</published>"
            + many_author_tags
            + "</entry></feed>"
        )
        with patch("urllib.request.urlopen", return_value=_urlopen_ctx(feed)):
            result = arxiv_search_impl("test")
        assert "+3 more" in result


# ---------------------------------------------------------------------------
# GitHub search
# ---------------------------------------------------------------------------


class TestGithubSearch:
    _REPO_RESPONSE = json.dumps(
        {
            "total_count": 42,
            "items": [
                {
                    "full_name": "openai/openai-python",
                    "description": "The official Python library for the OpenAI API",
                    "stargazers_count": 12000,
                    "language": "Python",
                    "html_url": "https://github.com/openai/openai-python",
                }
            ],
        }
    )

    _CODE_RESPONSE = json.dumps(
        {
            "total_count": 5,
            "items": [
                {
                    "repository": {"full_name": "owner/repo"},
                    "path": "src/main.py",
                    "html_url": "https://github.com/owner/repo/blob/main/src/main.py",
                }
            ],
        }
    )

    def test_repo_search(self):
        with patch("urllib.request.urlopen", return_value=_urlopen_ctx(self._REPO_RESPONSE)):
            result = github_search_impl("openai python")
        assert "openai/openai-python" in result
        assert "12,000" in result
        assert "Python" in result

    def test_code_search(self):
        with patch("urllib.request.urlopen", return_value=_urlopen_ctx(self._CODE_RESPONSE)):
            result = github_search_impl("def my_func", kind="code")
        assert "owner/repo" in result
        assert "src/main.py" in result

    def test_invalid_kind(self):
        result = github_search_impl("test", kind="issues")
        assert "must be" in result.lower()

    def test_rate_limit_error(self):
        import urllib.error

        err = urllib.error.HTTPError(url="", code=403, msg="Forbidden", hdrs=None, fp=None)  # type: ignore[arg-type]
        with patch("urllib.request.urlopen", side_effect=err):
            result = github_search_impl("test")
        assert "rate limit" in result.lower() or "GITHUB_TOKEN" in result

    def test_uses_token_from_env(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_testtoken")
        captured: list[dict] = []

        def fake_urlopen(req, timeout=None):
            captured.append(dict(req.headers))
            return _urlopen_ctx(self._REPO_RESPONSE)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            github_search_impl("test")
        assert any("Bearer ghp_testtoken" in str(v) for h in captured for v in h.values())


# ---------------------------------------------------------------------------
# GitHub read file
# ---------------------------------------------------------------------------


class TestGithubReadFile:
    _FILE_RESPONSE = json.dumps(
        {
            "encoding": "base64",
            "content": "aGVsbG8gd29ybGQ=\n",  # "hello world"
            "sha": "abc1234567890",
            "path": "README.md",
        }
    )

    def test_reads_text_file_shorthand(self):
        with patch("urllib.request.urlopen", return_value=_urlopen_ctx(self._FILE_RESPONSE)):
            result = github_read_file_impl("owner/repo/README.md")
        assert "hello world" in result
        assert "abc1234" in result

    def test_reads_from_blob_url(self):
        url = "https://github.com/owner/repo/blob/main/README.md"
        with patch("urllib.request.urlopen", return_value=_urlopen_ctx(self._FILE_RESPONSE)):
            result = github_read_file_impl(url)
        assert "hello world" in result

    def test_not_found(self):
        import urllib.error

        err = urllib.error.HTTPError(url="", code=404, msg="Not Found", hdrs=None, fp=None)  # type: ignore[arg-type]
        with patch("urllib.request.urlopen", side_effect=err):
            result = github_read_file_impl("owner/repo/missing.py")
        assert "not found" in result.lower()

    def test_invalid_shorthand(self):
        result = github_read_file_impl("not-enough-parts")
        assert "owner/repo" in result.lower() or "shorthand" in result.lower()

    def test_directory_response(self):
        dir_data = json.dumps(
            [
                {"name": "README.md", "type": "file"},
                {"name": "src", "type": "dir"},
            ]
        )
        with patch("urllib.request.urlopen", return_value=_urlopen_ctx(dir_data)):
            result = github_read_file_impl("owner/repo/src")
        assert "directory" in result.lower()

    def test_uses_token_from_env(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_testtoken2")
        captured: list[dict] = []

        def fake_urlopen(req, timeout=None):
            captured.append(dict(req.headers))
            return _urlopen_ctx(self._FILE_RESPONSE)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            github_read_file_impl("owner/repo/file.py")
        assert any("ghp_testtoken2" in str(v) for h in captured for v in h.values())


# ---------------------------------------------------------------------------
# Wikipedia
# ---------------------------------------------------------------------------


class TestWikipedia:
    _SUMMARY = json.dumps(
        {
            "type": "standard",
            "title": "Python (programming language)",
            "description": "High-level programming language",
            "extract": "Python is a high-level, general-purpose programming language.",
            "content_urls": {
                "desktop": {"page": "https://en.wikipedia.org/wiki/Python_(programming_language)"}
            },
        }
    )
    _FULL_API = json.dumps(
        {
            "query": {
                "pages": {
                    "123": {
                        "title": "Python (programming language)",
                        "extract": "Python is a high-level language. " * 100,
                    }
                }
            }
        }
    )

    def test_returns_summary(self):
        with patch("urllib.request.urlopen", return_value=_urlopen_ctx(self._SUMMARY)):
            result = wikipedia_impl("Python")
        assert "Python" in result
        assert "programming language" in result

    def test_disambiguation_page(self):
        disambiguation = json.dumps(
            {
                "type": "disambiguation",
                "title": "Python",
                "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Python"}},
            }
        )
        with patch("urllib.request.urlopen", return_value=_urlopen_ctx(disambiguation)):
            result = wikipedia_impl("Python")
        assert "disambiguation" in result.lower()

    def test_not_found(self):
        import urllib.error

        err = urllib.error.HTTPError(url="", code=404, msg="Not Found", hdrs=None, fp=None)  # type: ignore[arg-type]
        with patch("urllib.request.urlopen", side_effect=err):
            result = wikipedia_impl("NonExistentXYZ")
        assert "not found" in result.lower()

    def test_full_mode(self):
        responses = [_urlopen_ctx(self._SUMMARY), _urlopen_ctx(self._FULL_API)]
        with patch("urllib.request.urlopen", side_effect=responses):
            result = wikipedia_impl("Python", full=True)
        assert "Python is a high-level language" in result

    def test_lang_parameter(self):
        """Non-English language codes should be embedded in the request URL."""
        captured: list[str] = []

        def fake_urlopen(req, timeout=None):
            captured.append(req.full_url)
            return _urlopen_ctx(self._SUMMARY)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            wikipedia_impl("Python", lang="de")
        assert any("de.wikipedia.org" in u for u in captured)


# ---------------------------------------------------------------------------
# YouTube transcript
# ---------------------------------------------------------------------------


class TestExtractVideoId:
    def test_watch_url(self):
        assert _extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_short_url(self):
        assert _extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_bare_id(self):
        assert _extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"


class TestYoutubeTranscript:
    def _make_transcript_api(self, segments: list[dict]):
        """Build a mock YouTubeTranscriptApi tree."""
        seg_obj = MagicMock()
        seg_obj.__iter__ = MagicMock(return_value=iter(segments))

        transcript = MagicMock()
        transcript.fetch.return_value = segments
        transcript.language_code = "en"

        transcript_list = MagicMock()
        transcript_list.find_transcript.return_value = transcript

        api_cls = MagicMock()
        api_cls.list_transcripts.return_value = transcript_list
        return api_cls

    def test_returns_transcript(self):
        segments = [
            {"start": 0.0, "duration": 3.0, "text": "Hello world"},
            {"start": 60.0, "duration": 3.0, "text": "Another segment"},
        ]
        api = self._make_transcript_api(segments)
        with patch.dict(
            "sys.modules",
            {
                "youtube_transcript_api": MagicMock(
                    YouTubeTranscriptApi=api,
                    NoTranscriptFound=Exception,
                    TranscriptsDisabled=Exception,
                    VideoUnavailable=Exception,
                )
            },
        ):
            result = youtube_transcript_impl("dQw4w9WgXcQ")
        assert "dQw4w9WgXcQ" in result
        assert "Hello world" in result
        assert "Another segment" in result
        assert "[00:00]" in result

    def test_accepts_full_url(self):
        segments = [{"start": 0.0, "duration": 5.0, "text": "Hi"}]
        api = self._make_transcript_api(segments)
        with patch.dict(
            "sys.modules",
            {
                "youtube_transcript_api": MagicMock(
                    YouTubeTranscriptApi=api,
                    NoTranscriptFound=Exception,
                    TranscriptsDisabled=Exception,
                    VideoUnavailable=Exception,
                )
            },
        ):
            result = youtube_transcript_impl("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert "dQw4w9WgXcQ" in result

    def test_graceful_when_not_installed(self):
        with patch.dict("sys.modules", {"youtube_transcript_api": None}):
            result = youtube_transcript_impl("dQw4w9WgXcQ")
        assert "not installed" in result.lower() or "pip install" in result

    def test_disabled_transcripts(self):
        class TranscriptsDisabled(Exception):
            pass

        api = MagicMock()
        api.list_transcripts.side_effect = TranscriptsDisabled()
        with patch.dict(
            "sys.modules",
            {
                "youtube_transcript_api": MagicMock(
                    YouTubeTranscriptApi=api,
                    NoTranscriptFound=Exception,
                    TranscriptsDisabled=TranscriptsDisabled,
                    VideoUnavailable=Exception,
                )
            },
        ):
            result = youtube_transcript_impl("dQw4w9WgXcQ")
        assert "disabled" in result.lower()

    def test_truncates_long_transcript(self):
        segments = [{"start": float(i * 2), "duration": 2.0, "text": "word " * 100}
                    for i in range(100)]
        api = self._make_transcript_api(segments)
        with patch.dict(
            "sys.modules",
            {
                "youtube_transcript_api": MagicMock(
                    YouTubeTranscriptApi=api,
                    NoTranscriptFound=Exception,
                    TranscriptsDisabled=Exception,
                    VideoUnavailable=Exception,
                )
            },
        ):
            result = youtube_transcript_impl("dQw4w9WgXcQ", max_chars=200)
        assert "[truncated]" in result


# ---------------------------------------------------------------------------
# screenshot_multi_impl
# ---------------------------------------------------------------------------


class TestScreenshotMulti:
    def test_no_playwright(self):
        with patch.dict("sys.modules", {"playwright": None, "playwright.sync_api": None}):
            result = screenshot_multi_impl(["https://example.com"])
        assert "playwright" in result.lower()

    def test_empty_urls(self):
        # Returns early before touching playwright
        result = screenshot_multi_impl([])
        assert "No URLs" in result

    def test_multiple_urls_stitched(self, tmp_path):
        """Playwright + Pillow fully mocked; composite path should appear in output."""
        # Build playwright mock hierarchy
        mock_page = MagicMock()

        def fake_screenshot(path, **kw):
            open(path, "wb").write(b"FAKEPNG")

        mock_page.screenshot = MagicMock(side_effect=fake_screenshot)
        mock_browser = MagicMock()
        mock_browser.new_page.return_value = mock_page
        mock_pw_ctx = MagicMock()
        mock_pw_ctx.chromium.launch.return_value = mock_browser

        mock_pw_cm = MagicMock()
        mock_pw_cm.__enter__ = lambda s: mock_pw_ctx
        mock_pw_cm.__exit__ = MagicMock(return_value=False)
        mock_sync_playwright = MagicMock(return_value=mock_pw_cm)

        # Build Pillow mock
        mock_img = MagicMock()
        mock_img.width = 1280
        mock_img.height = 800
        mock_image_mod = MagicMock()
        mock_image_mod.open.return_value = mock_img
        mock_image_mod.new.return_value = mock_img

        fake_pw_module = MagicMock()
        fake_pw_module.sync_playwright = mock_sync_playwright
        fake_pw_module.TimeoutError = TimeoutError

        fake_pil = MagicMock()
        fake_pil.Image = mock_image_mod

        with patch.dict(
            "sys.modules",
            {
                "playwright": fake_pw_module,
                "playwright.sync_api": fake_pw_module,
                "PIL": fake_pil,
                "PIL.Image": mock_image_mod,
            },
        ):
            result = screenshot_multi_impl(
                ["https://a.com", "https://b.com"],
                save_dir=str(tmp_path),
            )
        assert "composite" in result.lower() or "screenshot" in result.lower()
