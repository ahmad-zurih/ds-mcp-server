"""
Web Tools: fetch webpages, search the web, and screenshot pages.
Gives the AI agent internet access for research and site cloning.
"""

import os
import re
import urllib.error
import urllib.request


def fetch_webpage_impl(url: str) -> str:
    """
    Fetch a URL and return structured content: title, meta description, navigation,
    headings, color palette, font families, and main page text (up to 4000 chars).
    Ideal for understanding a site layout and design before cloning or referencing it.
    """
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
                )
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read(600_000).decode("utf-8", errors="replace")

        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(raw, "html.parser")
            for tag in soup(["script", "noscript", "svg", "iframe"]):
                tag.decompose()

            title = soup.title.string.strip() if soup.title else "No title"
            meta_desc = ""
            meta = soup.find("meta", attrs={"name": "description"}) or soup.find(
                "meta", attrs={"property": "og:description"}
            )
            if meta:
                meta_desc = (meta.get("content") or "")[:200]

            nav_links: list[str] = []
            for nav in soup.find_all("nav"):
                for a in nav.find_all("a", href=True):
                    text = a.get_text(strip=True)
                    if text and len(text) < 50:
                        nav_links.append(text)

            headings: list[str] = []
            for tag in soup.find_all(["h1", "h2", "h3"]):
                text = tag.get_text(strip=True)[:100]
                if text:
                    headings.append("  " + tag.name.upper() + ": " + text)

            colors: set[str] = set()
            for style_tag in soup.find_all("style"):
                body = style_tag.get_text() or ""
                color_pat = r"#[0-9a-fA-F]{3,8}|rgb[(][^)]+[)]|rgba[(][^)]+[)]|hsl[(][^)]+[)]"
                for m in re.findall(color_pat, body):
                    colors.add(m)
            for tag in soup.find_all(style=True):
                color_pat2 = r"#[0-9a-fA-F]{3,8}|rgb[(][^)]+[)]|rgba[(][^)]+[)]"
                for m in re.findall(color_pat2, tag["style"]):
                    colors.add(m)

            fonts: set[str] = set()
            for style_tag in soup.find_all("style"):
                body = style_tag.get_text() or ""
                font_pat = r"font-family\s*:\s*([^;{}]+)"
                for m in re.findall(font_pat, body):
                    clean = m.strip().strip(chr(34) + chr(39))[:60]
                    if clean:
                        fonts.add(clean)

            main_text = soup.get_text(separator="\n", strip=True)
            newline_pat = r"\n{3,}"
            main_text = re.sub(newline_pat, "\n\n", main_text)
            if len(main_text) > 4000:
                main_text = main_text[:4000] + "\n... [truncated]"

        except ImportError:
            title_pat = r"<title[^>]*>(.*?)</title>"
            title_m = re.search(title_pat, raw, re.IGNORECASE | re.DOTALL)
            title = title_m.group(1).strip() if title_m else "No title"
            meta_desc, nav_links, headings = "", [], []
            colors, fonts = set(), set()
            text = re.sub(r"<[^>]+>", " ", raw)
            ws_pat = r"\s+"
            main_text = re.sub(ws_pat, " ", text)[:4000]

        parts = ["URL: " + url, "Title: " + title]
        if meta_desc:
            parts.append("Meta description: " + meta_desc)
        if nav_links:
            parts.append("Navigation items: " + " | ".join(nav_links[:20]))
        if headings:
            parts.append("Page headings:\n" + "\n".join(headings[:25]))
        if colors:
            parts.append("Color palette (CSS): " + ", ".join(sorted(colors)[:25]))
        if fonts:
            parts.append("Font families: " + ", ".join(sorted(fonts)[:10]))
        parts.append("Page content:\n" + main_text)
        return "\n".join(parts)

    except urllib.error.URLError as e:
        return "Error fetching " + url + ": " + str(e.reason)
    except Exception as e:
        return "Error fetching " + url + ": " + str(e)


def search_web_impl(query: str, max_results: int = 5) -> str:
    """Search the internet using DuckDuckGo. No API key required."""
    try:
        from ddgs import DDGS
    except ImportError:
        return "Error: ddgs not installed. Run: pip install ddgs"
    try:
        results: list[str] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    "Title:   " + r.get("title", "").strip() + "\n"
                    "URL:     " + r.get("href", "").strip() + "\n"
                    "Snippet: " + r.get("body", "").strip()[:300]
                )
        if not results:
            return "No results found for: " + query
        return "Search results for: " + query + "\n\n" + "\n\n---\n\n".join(results)
    except Exception as e:
        return "Search error: " + str(e)


def screenshot_webpage_impl(url: str, save_path: str | None = None) -> str:
    """
    Take a 1440x900 viewport screenshot using headless Chromium.
    Returns the path of the saved PNG file.
    Requires: pip install playwright && playwright install chromium
    """
    try:
        from playwright.sync_api import TimeoutError as PWTimeout
        from playwright.sync_api import sync_playwright
    except ImportError:
        return (
            "playwright not installed. Run:\n"
            "  pip install playwright\n"
            "  playwright install chromium"
        )
    try:
        if not save_path:
            safe = re.sub(r"[^\w.-]", "_", url.split("//")[-1][:55]).strip("_")
            screenshots_dir = os.path.join(
                os.path.expanduser("~"), "mcp-server-example", "screenshots"
            )
            os.makedirs(screenshots_dir, exist_ok=True)
            save_path = os.path.join(screenshots_dir, safe + ".png")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=25_000)
                page.wait_for_timeout(1500)
            except PWTimeout:
                pass
            page.screenshot(path=save_path, full_page=False)
            browser.close()

        return "Screenshot saved to: " + save_path
    except Exception as e:
        return "Screenshot error: " + str(e)


def screenshot_multi_impl(
    urls: list[str],
    layout: str = "side-by-side",
    save_dir: str | None = None,
) -> str:
    """
    Screenshot multiple webpages and stitch them into a single composite PNG.

    urls:     list of URLs to capture (capped at 6).
    layout:   'side-by-side' (horizontal, default) or 'vertical' (stacked).
    save_dir: directory for output files; auto-generated if omitted.

    Requires playwright (pip install playwright && playwright install chromium).
    Uses Pillow for stitching if available; falls back to returning individual paths.
    """
    if not urls:
        return "No URLs provided."

    try:
        from playwright.sync_api import TimeoutError as PWTimeout
        from playwright.sync_api import sync_playwright
    except ImportError:
        return (
            "playwright not installed. Run:\n"
            "  pip install playwright\n"
            "  playwright install chromium"
        )

    out_dir = save_dir or os.path.join(
        os.path.expanduser("~"), "mcp-server-example", "screenshots"
    )
    os.makedirs(out_dir, exist_ok=True)

    screenshots: list[str] = []
    errors: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        for url in urls[:6]:
            safe = re.sub(r"[^\w.-]", "_", url.split("//")[-1][:55]).strip("_")
            path = os.path.join(out_dir, f"multi_{safe}.png")
            try:
                page = browser.new_page(viewport={"width": 1280, "height": 800})
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=20_000)
                    page.wait_for_timeout(1200)
                except PWTimeout:
                    pass
                page.screenshot(path=path, full_page=False)
                page.close()
                screenshots.append(path)
            except Exception as exc:
                errors.append(f"{url}: {exc}")
        browser.close()

    if not screenshots:
        return "All screenshots failed: " + "; ".join(errors)

    # Stitch with Pillow when available
    try:
        from PIL import Image  # type: ignore[import]

        images = [Image.open(p) for p in screenshots]
        if layout == "vertical":
            total_h = sum(img.height for img in images)
            max_w = max(img.width for img in images)
            canvas = Image.new("RGB", (max_w, total_h), (28, 28, 38))
            y = 0
            for img in images:
                canvas.paste(img, (0, y))
                y += img.height
        else:  # side-by-side
            total_w = sum(img.width for img in images)
            max_h = max(img.height for img in images)
            canvas = Image.new("RGB", (total_w, max_h), (28, 28, 38))
            x = 0
            for img in images:
                canvas.paste(img, (x, 0))
                x += img.width
        composite = os.path.join(out_dir, "multi_composite.png")
        canvas.save(composite)
        result = (
            f"Composite screenshot saved to: {composite}\n"
            f"Layout: {layout}, {len(screenshots)} page(s)\n"
        )
        if errors:
            result += "Failures: " + "; ".join(errors) + "\n"
        result += "Individual screenshots:\n" + "\n".join(
            f"  {p}" for p in screenshots
        )
        return result
    except ImportError:
        # Pillow not installed — return paths to individual screenshots
        result = f"{len(screenshots)} screenshot(s) saved:\n" + "\n".join(
            f"  {p}" for p in screenshots
        )
        if errors:
            result += "\nFailures: " + "; ".join(errors)
        return result
