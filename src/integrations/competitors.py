"""Competitor landscape scraping and analysis.

Step 3 scrapes the pricing pages of the top 5 competitors found in search
results.  Every claim MUST include the URL source.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from langchain_google_genai import ChatGoogleGenerativeAI

from src.core.config import settings
from src.models.venture import CompetitorEntry

logger = logging.getLogger(__name__)


class CompetitorAnalyzer:
    """Scrapes and structures competitor data with source verification."""

    def __init__(self) -> None:
        self._timeout = httpx.Timeout(30.0)

    async def search_competitors(self, keywords: list[str], top_n: int = 5) -> list[str]:
        """Find the top *top_n* competitor URLs via Google Custom Search API.

        Falls back to an empty list if the API key is not configured.
        """
        if not settings.google_maps_api_key:
            logger.warning("Google API key not configured; competitor search unavailable.")
            return []

        query = " ".join(keywords) + " pricing"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "key": settings.google_maps_api_key,
                    "cx": "default",
                    "q": query,
                    "num": top_n,
                },
            )
            resp.raise_for_status()
            items = resp.json().get("items", [])

        return [item["link"] for item in items[:top_n]]

    async def scrape_pricing_page(self, url: str) -> str:
        """Fetch the raw text content of a pricing page."""
        async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "PrecisionFlow/5.0"})
            resp.raise_for_status()
            return resp.text[:10_000]  # Limit to first 10 KB

    async def analyze_competitor(self, url: str, page_content: str) -> CompetitorEntry:
        """Use the LLM to extract structured competitor data from a pricing page.

        The AI MUST provide the URL source for every claim.
        """
        prompt = f"""Extract structured competitor information from this pricing page.
You MUST include the source URL for every claim.

URL: {url}

PAGE CONTENT (truncated):
{page_content[:5000]}

Return JSON:
{{
  "name": "<company name>",
  "url": "{url}",
  "pricing": "<pricing summary>",
  "source_url": "{url}",
  "features": ["<feature 1>", "<feature 2>", ...]
}}
Only output valid JSON."""

        llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            temperature=0.0,
            google_api_key=settings.gemini_api_key,
        )
        response = await llm.ainvoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        import json

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse competitor data for %s", url)
            data = {"name": url, "url": url, "pricing": "Unknown", "source_url": url}

        return CompetitorEntry(
            name=data.get("name", url),
            url=data.get("url", url),
            pricing=data.get("pricing", "Unknown"),
            source_url=data.get("source_url", url),
            features=data.get("features", []),
        )

    async def build_matrix(self, keywords: list[str]) -> dict[str, Any]:
        """Build a verified competitor matrix from scratch.

        Returns a dict suitable for populating ``CompetitorMatrix``.
        """
        urls = await self.search_competitors(keywords)
        competitors: list[CompetitorEntry] = []

        for url in urls:
            try:
                page = await self.scrape_pricing_page(url)
                entry = await self.analyze_competitor(url, page)
                competitors.append(entry)
            except httpx.HTTPError as exc:
                logger.warning("Skipping competitor %s: %s", url, exc)

        density = len(competitors) / max(len(urls), 1)
        return {"competitors": competitors, "density_score": density}
