"""Market data integrations — real APIs, not AI estimates.

Step 2 uses these services to fetch hard search-volume data.
If search volume == 0, the system LOCKS and warns the user.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from src.core.config import settings

logger = logging.getLogger(__name__)


class MarketDataClient:
    """Aggregates market signals from Google Trends, Semrush, and Reddit."""

    def __init__(self) -> None:
        self._timeout = httpx.Timeout(30.0)

    # ----- Google Trends -------------------------------------------------

    async def get_google_trends(self, keywords: list[str]) -> dict[str, Any]:
        """Fetch interest-over-time data for *keywords* via the Google Trends API."""
        if not settings.google_trends_api_key:
            logger.warning("GOOGLE_TRENDS_API_KEY not configured; skipping.")
            return {}

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(
                "https://trends.googleapis.com/trends/api/explore",
                params={"key": settings.google_trends_api_key, "q": ",".join(keywords)},
            )
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]

    # ----- Semrush -------------------------------------------------------

    async def get_search_volumes(self, keywords: list[str]) -> dict[str, int]:
        """Fetch monthly search volumes per keyword from Semrush."""
        if not settings.semrush_api_key:
            logger.warning("SEMRUSH_API_KEY not configured; skipping.")
            return {}

        volumes: dict[str, int] = {}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for kw in keywords:
                response = await client.get(
                    "https://api.semrush.com/",
                    params={
                        "type": "phrase_this",
                        "key": settings.semrush_api_key,
                        "phrase": kw,
                        "database": "us",
                    },
                )
                response.raise_for_status()
                # Semrush returns CSV; second line has the volume.
                lines = response.text.strip().split("\n")
                if len(lines) >= 2:
                    parts = lines[1].split(";")
                    volumes[kw] = int(parts[2]) if len(parts) > 2 else 0
                else:
                    volumes[kw] = 0
        return volumes

    # ----- Reddit --------------------------------------------------------

    async def get_reddit_mentions(self, keywords: list[str]) -> int:
        """Count recent Reddit posts/comments mentioning any of *keywords*."""
        if not settings.reddit_client_id:
            logger.warning("REDDIT_CLIENT_ID not configured; skipping.")
            return 0

        # Authenticate via OAuth2
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            auth_resp = await client.post(
                "https://www.reddit.com/api/v1/access_token",
                auth=(settings.reddit_client_id, settings.reddit_client_secret),
                data={"grant_type": "client_credentials"},
                headers={"User-Agent": "PrecisionFlow/5.0"},
            )
            auth_resp.raise_for_status()
            token = auth_resp.json()["access_token"]

            query = " OR ".join(keywords)
            search_resp = await client.get(
                "https://oauth.reddit.com/search.json",
                params={"q": query, "limit": 100, "sort": "relevance", "t": "month"},
                headers={
                    "Authorization": f"Bearer {token}",
                    "User-Agent": "PrecisionFlow/5.0",
                },
            )
            search_resp.raise_for_status()
            data = search_resp.json()

        children = data.get("data", {}).get("children", [])
        return len(children)

    # ----- Aggregated report ---------------------------------------------

    async def build_report(
        self, keywords: list[str]
    ) -> dict[str, Any]:
        """Build a complete market data report from all sources.

        Returns a dict suitable for populating ``MarketDataReport``.
        """
        search_volumes = await self.get_search_volumes(keywords)
        trend_data = await self.get_google_trends(keywords)
        reddit_mentions = await self.get_reddit_mentions(keywords)

        total_volume = sum(search_volumes.values())
        has_valid_data = total_volume > 0

        sources: list[str] = []
        if search_volumes:
            sources.append("Semrush")
        if trend_data:
            sources.append("Google Trends")
        if reddit_mentions > 0:
            sources.append("Reddit")

        return {
            "keywords": keywords,
            "search_volumes": search_volumes,
            "trend_data": trend_data,
            "reddit_mentions": reddit_mentions,
            "data_sources": sources,
            "has_valid_data": has_valid_data,
        }
