# scraper.py - Web scraping utilities

import requests
from bs4 import BeautifulSoup
import re


def scrape_url(url: str, timeout: int = 15) -> str:
    """Scrape and extract main content from a URL."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = "\n".join(lines)

        # Truncate to avoid huge texts
        max_chars = 5000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[...truncated]"

        return text if text else "No content could be extracted."

    except Exception as e:
        return f"Error scraping {url}: {e}"


def try_trafilatura(url: str) -> str:
    """Extract content using Trafilatura (better for news/articles)."""
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            if text:
                max_chars = 5000
                if len(text) > max_chars:
                    text = text[:max_chars] + "\n\n[...truncated]"
                return text
        return scrape_url(url)
    except ImportError:
        return scrape_url(url)
    except Exception as e:
        return f"Error with trafilatura on {url}: {e}"
