"""
Research & reference tools: arXiv, GitHub, Wikipedia, YouTube transcripts.
All HTTP calls use stdlib urllib so there are no mandatory extra dependencies
(youtube-transcript-api is optional and imported at call-time).
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

# ---------------------------------------------------------------------------
# Shared HTTP helpers
# ---------------------------------------------------------------------------

_USER_AGENT = (
    "ds-mcp-server/research (+https://github.com/ahmad-zurih/ds-mcp-server)"
)
_DEFAULT_HEADERS: dict[str, str] = {
    "User-Agent": _USER_AGENT,
    "Accept": "application/json",
}


def _get_json(url: str, headers: dict[str, str] | None = None) -> Any:
    h = {**_DEFAULT_HEADERS, **(headers or {})}
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_text(url: str, headers: dict[str, str] | None = None) -> str:
    h = {**_DEFAULT_HEADERS, "Accept": "text/plain", **(headers or {})}
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------

_ATOM_NS = "http://www.w3.org/2005/Atom"


def arxiv_search_impl(query: str, max_results: int = 5) -> str:
    """Search arXiv for papers matching the query (free API, no key required)."""
    n = min(max(1, max_results), 20)
    params = urllib.parse.urlencode(
        {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": n,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )
    url = f"http://export.arxiv.org/api/query?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8")
    except Exception as exc:
        return f"arXiv search error: {exc}"

    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        return f"arXiv response parse error: {exc}"

    ns = {"a": _ATOM_NS}
    entries = root.findall("a:entry", ns)
    if not entries:
        return f"No arXiv results found for: {query!r}"

    parts: list[str] = [
        f"arXiv search results for: {query!r}  ({len(entries)} returned)\n"
    ]
    for i, entry in enumerate(entries, 1):

        def _txt(tag: str) -> str:
            el = entry.find(f"a:{tag}", ns)
            return (el.text or "").strip() if el is not None else ""

        arxiv_id = _txt("id").split("/abs/")[-1]
        title = re.sub(r"\s+", " ", _txt("title"))
        summary = re.sub(r"\s+", " ", _txt("summary"))
        if len(summary) > 400:
            summary = summary[:400] + "..."
        published = _txt("published")[:10]

        authors = [
            a.find("a:name", ns).text.strip()
            for a in entry.findall("a:author", ns)
            if a.find("a:name", ns) is not None
        ]
        author_str = ", ".join(authors[:5])
        if len(authors) > 5:
            author_str += f" +{len(authors)-5} more"

        pdf_link = ""
        for link in entry.findall("a:link", ns):
            if link.get("type") == "application/pdf":
                pdf_link = link.get("href", "")
                break
        if not pdf_link and arxiv_id:
            pdf_link = f"https://arxiv.org/pdf/{arxiv_id}"

        base_id = arxiv_id.split("v")[0]
        parts.append(
            f"{i}. {title}\n"
            f"   ID:        {arxiv_id}\n"
            f"   Authors:   {author_str}\n"
            f"   Published: {published}\n"
            f"   Abstract:  {summary}\n"
            f"   PDF:       {pdf_link}\n"
            f"   Page:      https://arxiv.org/abs/{base_id}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# GitHub
# ---------------------------------------------------------------------------


def _github_headers() -> dict[str, str]:
    h = {**_DEFAULT_HEADERS, "X-GitHub-Api-Version": "2022-11-28"}
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def github_search_impl(
    query: str, kind: str = "repos", max_results: int = 5
) -> str:
    """
    Search GitHub repositories or code.
    kind: 'repos' (default) or 'code'.
    Rate-limited without GITHUB_TOKEN (10 req/hr); set GITHUB_TOKEN for 30/min.
    """
    kind = kind.lower().strip()
    if kind not in ("repos", "code"):
        return "kind must be 'repos' or 'code'."
    n = min(max(1, max_results), 30)
    params = urllib.parse.urlencode({"q": query, "per_page": n})
    url = f"https://api.github.com/search/{kind}?{params}"
    try:
        data = _get_json(url, headers=_github_headers())
    except urllib.error.HTTPError as exc:
        if exc.code == 403:
            return (
                "GitHub rate limit exceeded. Set the GITHUB_TOKEN environment variable "
                "with a Personal Access Token to raise limits (30 requests/min)."
            )
        if exc.code == 422:
            return "GitHub rejected the query. Try a simpler search term."
        return f"GitHub API error {exc.code}: {exc.reason}"
    except Exception as exc:
        return f"GitHub search error: {exc}"

    items = data.get("items", [])
    total = data.get("total_count", 0)
    if not items:
        return f"No GitHub {kind} results for: {query!r}"

    parts: list[str] = [
        f"GitHub {kind} search: {query!r}  ({len(items)} of {total:,})\n"
    ]
    if kind == "repos":
        for i, item in enumerate(items, 1):
            stars = item.get("stargazers_count", 0)
            lang = item.get("language") or "—"
            desc = (item.get("description") or "").strip()[:120]
            parts.append(
                f"{i}. {item['full_name']}\n"
                f"   ⭐ {stars:,}  Language: {lang}\n"
                f"   {desc}\n"
                f"   {item.get('html_url', '')}"
            )
    else:
        for i, item in enumerate(items, 1):
            repo = item.get("repository", {}).get("full_name", "?")
            path = item.get("path", "?")
            html_url = item.get("html_url", "")
            parts.append(f"{i}. {repo} — {path}\n   {html_url}")
    return "\n\n".join(parts)


def github_read_file_impl(url_or_path: str, ref: str = "HEAD") -> str:
    """
    Read a file from a public GitHub repository.
    Accepts:
      • GitHub blob URL:  https://github.com/owner/repo/blob/main/path/file.py
      • Raw URL:          https://raw.githubusercontent.com/owner/repo/main/path/file.py
      • Shorthand:        owner/repo/path/to/file  (uses `ref` for branch/tag/SHA)
    Returns decoded file content, truncated at 8 000 characters.
    """
    import base64

    owner = repo = path = ""
    resolved_ref = ref

    blob_re = re.compile(r"github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)")
    raw_re = re.compile(
        r"raw\.githubusercontent\.com/([^/]+)/([^/]+)/([^/]+)/(.+)"
    )

    if m := blob_re.search(url_or_path):
        owner, repo, resolved_ref, path = m.groups()
    elif m := raw_re.search(url_or_path):
        owner, repo, resolved_ref, path = m.groups()
        raw_url = (
            url_or_path
            if url_or_path.startswith("http")
            else f"https://{url_or_path}"
        )
        try:
            text = _get_text(raw_url, headers=_github_headers())
        except Exception as exc:
            return f"Error reading {raw_url}: {exc}"
        if len(text) > 8000:
            text = text[:8000] + "\n... [truncated at 8 000 chars]"
        return text
    else:
        parts = url_or_path.lstrip("/").split("/", 2)
        if len(parts) < 3:
            return (
                "Provide a GitHub URL or 'owner/repo/path/to/file' shorthand. "
                f"Got: {url_or_path!r}"
            )
        owner, repo, path = parts

    ref_qs = (
        f"?ref={urllib.parse.quote(resolved_ref)}" if resolved_ref != "HEAD" else ""
    )
    api_url = (
        f"https://api.github.com/repos/{owner}/{repo}/contents/{path}{ref_qs}"
    )
    try:
        data = _get_json(api_url, headers=_github_headers())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return f"File not found: {owner}/{repo}/{path} @ {resolved_ref}"
        if exc.code == 403:
            return (
                "GitHub rate limit or permission error. "
                "Set GITHUB_TOKEN env var for higher limits."
            )
        return f"GitHub API error {exc.code}: {exc.reason}"
    except Exception as exc:
        return f"GitHub read error: {exc}"

    if isinstance(data, list):
        names = [
            ("/" if e.get("type") == "dir" else " ") + " " + e["name"]
            for e in data
        ]
        return f"{owner}/{repo}/{path} is a directory:\n" + "\n".join(names)

    encoding = data.get("encoding", "")
    if encoding != "base64":
        return f"Unexpected encoding {encoding!r} — try the raw URL instead."
    try:
        text = base64.b64decode(data.get("content", "")).decode(
            "utf-8", errors="replace"
        )
    except Exception:
        return "Could not decode file content (binary file?)."

    sha = data.get("sha", "")[:7]
    if len(text) > 8000:
        text = text[:8000] + "\n... [truncated at 8 000 chars]"
    return f"# {owner}/{repo}/{path} @ {sha}\n\n{text}"


# ---------------------------------------------------------------------------
# Wikipedia
# ---------------------------------------------------------------------------


def wikipedia_impl(title: str, lang: str = "en", full: bool = False) -> str:
    """
    Fetch a Wikipedia article by title.
    Returns a concise summary by default; pass full=True for the complete
    plain-text extract (up to 6 000 characters).
    lang: ISO language code, default 'en'.
    """
    encoded = urllib.parse.quote(title.replace(" ", "_"))
    summary_url = (
        f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{encoded}"
    )
    try:
        summary_data = _get_json(summary_url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return (
                f"Wikipedia article not found: {title!r}. "
                "Try a different spelling or language code."
            )
        return f"Wikipedia error {exc.code}: {exc.reason}"
    except Exception as exc:
        return f"Wikipedia error: {exc}"

    if summary_data.get("type") == "disambiguation":
        page_url = (
            summary_data.get("content_urls", {})
            .get("desktop", {})
            .get("page", "")
        )
        return (
            f"'{title}' is a disambiguation page on Wikipedia. "
            f"Please specify which meaning you want.\nSee: {page_url}"
        )

    page_title = summary_data.get("title", title)
    description = summary_data.get("description", "")
    extract = summary_data.get("extract", "")
    page_url = (
        summary_data.get("content_urls", {}).get("desktop", {}).get("page", "")
    )

    if not full:
        result = f"# {page_title}\n"
        if description:
            result += f"_{description}_\n\n"
        result += extract
        if page_url:
            result += f"\n\nFull article: {page_url}"
        return result

    # Full plain-text extract via action API
    api_url = (
        f"https://{lang}.wikipedia.org/w/api.php?"
        + urllib.parse.urlencode(
            {
                "action": "query",
                "titles": title,
                "prop": "extracts",
                "explaintext": "1",
                "exlimit": "1",
                "format": "json",
                "redirects": "1",
            }
        )
    )
    try:
        full_data = _get_json(api_url)
        pages = full_data.get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        full_text = page.get("extract", extract)
    except Exception:
        full_text = extract  # graceful fallback to summary extract

    if len(full_text) > 6000:
        full_text = full_text[:6000] + "\n... [truncated at 6 000 chars]"

    result = f"# {page_title}\n"
    if description:
        result += f"_{description}_\n\n"
    result += full_text
    if page_url:
        result += f"\n\nFull article: {page_url}"
    return result


# ---------------------------------------------------------------------------
# YouTube transcript
# ---------------------------------------------------------------------------


def _extract_video_id(url_or_id: str) -> str:
    """Extract an 11-character YouTube video ID from a URL or return it as-is."""
    patterns = [
        r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})",
        r"^([A-Za-z0-9_-]{11})$",
    ]
    for pat in patterns:
        if m := re.search(pat, url_or_id):
            return m.group(1)
    return url_or_id  # pass through; YouTubeTranscriptApi will give a clear error


def youtube_transcript_impl(
    url_or_id: str,
    languages: str = "en",
    max_chars: int = 8000,
) -> str:
    """
    Fetch the transcript of a YouTube video.
    url_or_id: full YouTube URL or bare 11-character video ID.
    languages:  comma-separated ISO codes in preference order, e.g. 'en,de,fr'.
    max_chars:  truncate output at this many characters (default 8 000).
    Requires:   pip install 'ds-mcp-server[research]'  (youtube-transcript-api).
    """
    try:
        from youtube_transcript_api import (  # type: ignore[import]
            NoTranscriptFound,
            TranscriptsDisabled,
            VideoUnavailable,
            YouTubeTranscriptApi,
        )
    except ImportError:
        return (
            "youtube-transcript-api is not installed.\n"
            "Install the research extra:  pip install 'ds-mcp-server[research]'\n"
            "or directly:                pip install youtube-transcript-api"
        )

    video_id = _extract_video_id(url_or_id)
    lang_list = [lc.strip() for lc in languages.split(",") if lc.strip()]

    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_transcript(lang_list)
        except NoTranscriptFound:
            transcript = transcript_list.find_generated_transcript(lang_list)
    except TranscriptsDisabled:
        return f"Transcripts are disabled for video: {video_id}"
    except VideoUnavailable:
        return f"Video is unavailable or private: {video_id}"
    except Exception as exc:
        return f"Could not fetch transcript for {video_id!r}: {exc}"

    try:
        segments = transcript.fetch()
    except Exception as exc:
        return f"Error fetching transcript segments: {exc}"

    # Group segments into ~60-second buckets with timestamps
    lines: list[str] = []
    bucket_start = 0.0
    bucket_words: list[str] = []
    for seg in segments:
        start = float(seg.get("start", 0))
        text = seg.get("text", "").strip().replace("\n", " ")
        if start - bucket_start >= 60 and bucket_words:
            mins, secs = divmod(int(bucket_start), 60)
            lines.append(f"[{mins:02d}:{secs:02d}] " + " ".join(bucket_words))
            bucket_words = []
            bucket_start = start
        bucket_words.append(text)
    if bucket_words:
        mins, secs = divmod(int(bucket_start), 60)
        lines.append(f"[{mins:02d}:{secs:02d}] " + " ".join(bucket_words))

    lang_code = getattr(transcript, "language_code", "?")
    header = (
        f"YouTube transcript — video: {video_id}  language: {lang_code}\n"
        f"URL: https://www.youtube.com/watch?v={video_id}\n\n"
    )
    body = "\n".join(lines)
    if len(body) > max_chars:
        body = body[:max_chars] + "\n... [truncated]"
    return header + body
