"""Golden Database — the proprietary vector store of 1,000+ proven business models.

The system does NOT ask the AI *"How do I price this?"*.  Instead, it
scans this database, finds the 3 most relevant case studies, and adapts
them.  It copies success rather than inventing fiction.

Backend: Supabase PostgreSQL + pgvector.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from pydantic import BaseModel, Field

from src.core.config import settings

logger = logging.getLogger(__name__)


class CaseStudy(BaseModel):
    """A single business case study stored in the Golden Database."""

    id: str = ""
    industry: str = ""
    business_model: str = ""
    company_name: str = ""
    summary: str = ""
    revenue_model: str = ""
    pricing_strategy: str = ""
    legal_structure: str = ""
    pitch_deck_url: str = ""
    embedding: list[float] = Field(default_factory=list, exclude=True)
    similarity_score: float = 0.0


class GoldenDatabase:
    """Client for the Supabase-backed vector store.

    All retrieval is similarity-based: the user's idea is embedded and the
    top-*k* nearest case studies are returned.
    """

    MATCH_FUNCTION = "match_case_studies"
    TABLE = "case_studies"

    def __init__(self) -> None:
        self._base_url = settings.supabase_url.rstrip("/")
        self._headers = {
            "apikey": settings.supabase_key,
            "Authorization": f"Bearer {settings.supabase_key}",
            "Content-Type": "application/json",
        }

    async def _rpc(self, function: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Call a Supabase RPC (Remote Procedure Call) function."""
        url = f"{self._base_url}/rest/v1/rpc/{function}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=self._headers, json=params)
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]

    async def search(self, query_embedding: list[float], top_k: int = 3) -> list[CaseStudy]:
        """Find the *top_k* most relevant case studies for a given embedding.

        Uses the ``match_case_studies`` Supabase RPC function backed by
        pgvector's cosine similarity index.
        """
        try:
            rows = await self._rpc(
                self.MATCH_FUNCTION,
                {"query_embedding": query_embedding, "match_count": top_k},
            )
        except httpx.HTTPError as exc:
            logger.error("Golden Database search failed: %s", exc)
            return []

        results: list[CaseStudy] = []
        for row in rows:
            results.append(
                CaseStudy(
                    id=str(row.get("id", "")),
                    industry=row.get("industry", ""),
                    business_model=row.get("business_model", ""),
                    company_name=row.get("company_name", ""),
                    summary=row.get("summary", ""),
                    revenue_model=row.get("revenue_model", ""),
                    pricing_strategy=row.get("pricing_strategy", ""),
                    legal_structure=row.get("legal_structure", ""),
                    pitch_deck_url=row.get("pitch_deck_url", ""),
                    similarity_score=float(row.get("similarity", 0.0)),
                )
            )
        return results

    async def insert(self, study: CaseStudy, embedding: list[float]) -> str:
        """Insert a new case study with its embedding vector."""
        url = f"{self._base_url}/rest/v1/{self.TABLE}"
        payload = study.model_dump()
        payload["embedding"] = embedding

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=self._headers, json=payload)
            response.raise_for_status()
            data = response.json()

        record_id = data[0]["id"] if isinstance(data, list) and data else ""
        logger.info("Inserted case study %s into Golden Database.", record_id)
        return str(record_id)

    async def get_embedding(self, text: str) -> list[float]:
        """Generate an embedding for arbitrary text via the OpenAI API."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": "text-embedding-3-small", "input": text},
            )
            response.raise_for_status()
            data = response.json()
        return data["data"][0]["embedding"]  # type: ignore[no-any-return]

    async def search_by_text(self, query: str, top_k: int = 3) -> list[CaseStudy]:
        """Convenience: embed *query* and search in one call."""
        embedding = await self.get_embedding(query)
        return await self.search(embedding, top_k=top_k)
