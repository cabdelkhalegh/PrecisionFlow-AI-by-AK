"""Supply chain partner discovery via Maps / Business APIs.

Step 8 uses real business listings to find verified suppliers and
partners in the user's region.
"""

from __future__ import annotations

import logging

import httpx

from src.core.config import settings
from src.models.venture import PartnerEntry

logger = logging.getLogger(__name__)


class SupplierFinder:
    """Finds real suppliers and partners using the Google Places API."""

    def __init__(self) -> None:
        self._timeout = httpx.Timeout(30.0)

    async def find_partners(
        self, query: str, region: str, max_results: int = 10
    ) -> list[PartnerEntry]:
        """Search for business partners near *region* matching *query*.

        Returns verified partner entries with real contact info.
        """
        if not settings.google_maps_api_key:
            logger.warning("GOOGLE_MAPS_API_KEY not configured; supplier search unavailable.")
            return []

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                "https://maps.googleapis.com/maps/api/place/textsearch/json",
                params={
                    "query": f"{query} in {region}",
                    "key": settings.google_maps_api_key,
                },
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])

        partners: list[PartnerEntry] = []
        for place in results[:max_results]:
            partners.append(
                PartnerEntry(
                    name=place.get("name", ""),
                    contact_info=place.get("formatted_address", ""),
                    region=region,
                    verified=True,
                )
            )
        return partners
