"""
Phase 1 — Wikipedia Scraper

Fetches a Wikipedia article, strips boilerplate, cleans the text, and
splits it into overlapping chunks suitable for downstream generation.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Generator

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class WikipediaScraper:
    """Fetch, clean, and chunk a Wikipedia article.

    Args:
        config: Full project config dict.
    """

    UNWANTED_SECTIONS = {
        "references", "see also", "further reading", "external links",
        "notes", "footnotes", "bibliography", "citations",
    }

    def __init__(self, config: dict) -> None:
        self.cfg = config["scraper"]
        self.paths = config["paths"]
        self.chunk_size: int = self.cfg["chunk_size"]
        self.chunk_overlap: int = self.cfg["chunk_overlap"]
        self.max_chunks: int = self.cfg["max_chunks"]
        self.retry_attempts: int = self.cfg["retry_attempts"]
        self.retry_delay: int = self.cfg["retry_delay"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, url: str | None = None) -> dict:
        """Execute the full scrape → clean → chunk pipeline.

        Args:
            url: Wikipedia URL to scrape. Falls back to config value.

        Returns:
            Dictionary with keys ``raw``, ``clean``, and ``chunks``.
        """
        url = url or self.cfg["wikipedia_url"]
        logger.info("Scraping URL: %s", url)

        raw_html = self._fetch_with_retry(url)
        raw_text = self._extract_text(raw_html)
        clean_text = self._clean_text(raw_text)
        chunks = list(self._chunk_text(clean_text))

        logger.info("Extracted %d chunks from %d characters", len(chunks), len(clean_text))

        self._save(raw_text, clean_text, chunks)
        return {"raw": raw_text, "clean": clean_text, "chunks": chunks}

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _fetch_with_retry(self, url: str) -> str:
        """GET *url* with exponential-back-off retries.

        Args:
            url: HTTP(S) URL to fetch.

        Returns:
            Raw HTML string.

        Raises:
            RuntimeError: After all retry attempts are exhausted.
        """
        headers = {"User-Agent": "Mozilla/5.0 (WikiScraper/1.0; educational)"}
        for attempt in range(1, self.retry_attempts + 1):
            try:
                logger.debug("Fetch attempt %d / %d", attempt, self.retry_attempts)
                resp = requests.get(url, headers=headers, timeout=30)
                resp.raise_for_status()
                return resp.text
            except requests.RequestException as exc:
                logger.warning("Attempt %d failed: %s", attempt, exc)
                if attempt < self.retry_attempts:
                    time.sleep(self.retry_delay * attempt)
        raise RuntimeError(f"Failed to fetch {url} after {self.retry_attempts} attempts")

    def _extract_text(self, html: str) -> str:
        """Parse HTML and return article body text only.

        Removes navigation, infoboxes, references, tables, and other
        Wikipedia boilerplate.

        Args:
            html: Raw HTML source.

        Returns:
            Extracted text string.
        """
        soup = BeautifulSoup(html, "lxml")

        # Remove non-content elements
        for tag in soup.find_all(["script", "style", "nav", "header", "footer",
                                   "table", "sup", "span.reference",
                                   "div.reflist", "div.navbox", "div.hatnote",
                                   "div.toc", "div.mw-editsection"]):
            tag.decompose()

        content_div = soup.find("div", {"id": "mw-content-text"}) or soup
        paragraphs: list[str] = []
        skip_section = False

        for element in content_div.find_all(["h2", "h3", "p"]):
            if element.name in ("h2", "h3"):
                heading = element.get_text().strip().lower().rstrip("[edit]").strip()
                skip_section = any(s in heading for s in self.UNWANTED_SECTIONS)
                continue
            if skip_section:
                continue
            text = element.get_text(" ", strip=True)
            if text:
                paragraphs.append(text)

        return "\n\n".join(paragraphs)

    @staticmethod
    def _clean_text(text: str) -> str:
        """Remove citations, excessive whitespace, and non-ASCII artefacts.

        Args:
            text: Raw extracted text.

        Returns:
            Cleaned text string.
        """
        # Remove citation markers like [1], [2], [note 3]
        text = re.sub(r"\[\d+\]|\[note \d+\]|\[citation needed\]", "", text)
        # Collapse whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        # Remove lone punctuation lines
        text = "\n".join(
            line for line in text.splitlines() if len(line.strip()) > 5
        )
        return text.strip()

    def _chunk_text(self, text: str) -> Generator[dict, None, None]:
        """Split *text* into overlapping word-based chunks.

        Args:
            text: Clean article text.

        Yields:
            Dicts with ``id``, ``text``, and ``word_count`` keys.
        """
        words = text.split()
        step = self.chunk_size - self.chunk_overlap
        chunk_id = 0

        for i in range(0, len(words), step):
            if chunk_id >= self.max_chunks:
                break
            chunk_words = words[i: i + self.chunk_size]
            if len(chunk_words) < 20:  # Skip tiny trailing fragments
                continue
            yield {
                "id": chunk_id,
                "text": " ".join(chunk_words),
                "word_count": len(chunk_words),
            }
            chunk_id += 1

    def _save(self, raw: str, clean: str, chunks: list[dict]) -> None:
        """Persist raw text, clean text, and chunks to disk.

        Args:
            raw: Raw extracted text.
            clean: Cleaned text.
            chunks: List of chunk dicts.
        """
        raw_path = Path(self.paths["data_raw"])
        raw_path.mkdir(parents=True, exist_ok=True)
        (raw_path / "raw_text.txt").write_text(raw, encoding="utf-8")

        proc_path = Path(self.paths["data_processed"])
        proc_path.mkdir(parents=True, exist_ok=True)
        (proc_path / "clean_text.txt").write_text(clean, encoding="utf-8")

        chunks_path = proc_path / "chunks.json"
        chunks_path.write_text(
            json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Saved raw text, clean text, and %d chunks.", len(chunks))
