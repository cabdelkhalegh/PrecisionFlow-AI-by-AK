#!/usr/bin/env python3
"""Golden Database Seeding Script

Populates the Supabase case_studies table with 1,000+ proven business models.

Usage:
    # Seed with anchor data only (~200 studies)
    python scripts/seed_golden_database.py

    # Expand to 1,000+ with AI generation
    python scripts/seed_golden_database.py --expand

    # Dry run (no database writes)
    python scripts/seed_golden_database.py --dry-run

    # Resume from specific index
    python scripts/seed_golden_database.py --start-index 150
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import httpx
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class GoldenDatabaseSeeder:
    """Handles batch seeding of the Golden Database with embeddings."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.base_url = settings.supabase_url.rstrip("/")
        self.headers = {
            "apikey": settings.supabase_key,
            "Authorization": f"Bearer {settings.supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }
        self.embedding_url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "text-embedding-004:embedContent"
        )

    async def get_embedding(self, text: str) -> list[float]:
        """Generate embedding via Gemini API."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self.embedding_url,
                params={"key": settings.gemini_api_key},
                headers={"Content-Type": "application/json"},
                json={
                    "model": "models/text-embedding-004",
                    "content": {"parts": [{"text": text}]},
                },
            )
            response.raise_for_status()
            data = response.json()
        return data["embedding"]["values"]

    async def batch_get_embeddings(
        self, texts: list[str], batch_size: int = 10
    ) -> list[list[float]]:
        """Generate embeddings in batches with rate limiting."""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = await asyncio.gather(
                *[self.get_embedding(text) for text in batch]
            )
            embeddings.extend(batch_embeddings)

            # Rate limiting: 10 requests per second
            if i + batch_size < len(texts):
                await asyncio.sleep(1.0)

        return embeddings

    async def insert_batch(self, records: list[dict[str, Any]]) -> int:
        """Insert a batch of case studies into Supabase."""
        if self.dry_run:
            logger.info("[DRY RUN] Would insert %d records", len(records))
            return len(records)

        url = f"{self.base_url}/rest/v1/case_studies"
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=self.headers, json=records)
            response.raise_for_status()
            inserted = response.json()

        logger.info("Inserted %d case studies", len(inserted))
        return len(inserted)

    def prepare_embedding_text(self, study: dict[str, Any]) -> str:
        """Create a rich text representation for embedding."""
        return (
            f"{study['industry']} - {study['business_model']}\n"
            f"Company: {study['company_name']}\n"
            f"Summary: {study['summary']}\n"
            f"Revenue: {study['revenue_model']}\n"
            f"Pricing: {study['pricing_strategy']}"
        )

    async def seed(
        self,
        case_studies: list[dict[str, Any]],
        batch_size: int = 50,
        start_index: int = 0,
    ) -> None:
        """Seed the database with case studies."""
        if start_index > 0:
            logger.info("Resuming from index %d", start_index)
            case_studies = case_studies[start_index:]

        total = len(case_studies)
        logger.info("Seeding %d case studies...", total)

        # Process in batches
        for i in range(0, total, batch_size):
            batch = case_studies[i : i + batch_size]
            logger.info(
                "Processing batch %d-%d of %d",
                i + start_index,
                min(i + batch_size, total) + start_index,
                total + start_index,
            )

            # Generate embeddings for this batch
            texts = [self.prepare_embedding_text(study) for study in batch]
            logger.info("Generating embeddings for %d studies...", len(batch))

            embeddings = await self.batch_get_embeddings(texts, batch_size=5)

            # Prepare records for insertion
            records = []
            for study, embedding in zip(batch, embeddings):
                record = {
                    "industry": study["industry"],
                    "business_model": study["business_model"],
                    "company_name": study["company_name"],
                    "summary": study["summary"],
                    "revenue_model": study.get("revenue_model", ""),
                    "pricing_strategy": study.get("pricing_strategy", ""),
                    "legal_structure": study.get("legal_structure", "Delaware C-Corp"),
                    "pitch_deck_url": study.get("pitch_deck_url", ""),
                    "embedding": embedding,
                }
                records.append(record)

            # Insert batch
            await self.insert_batch(records)
            logger.info("✓ Batch complete")

        logger.info("✅ Seeding complete! Total studies: %d", total + start_index)


async def expand_with_ai(
    anchor_studies: list[dict[str, Any]], target_count: int = 1000
) -> list[dict[str, Any]]:
    """Use Gemini to generate additional case studies based on anchor data.

    Takes the anchor studies and generates variations (different industries,
    geographies, scales, business models) to reach the target count.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    logger.info(
        "Expanding %d anchor studies to %d total...",
        len(anchor_studies),
        target_count,
    )

    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=0.7,  # More creative for variations
        google_api_key=settings.gemini_api_key,
    )

    expanded_studies = anchor_studies.copy()
    remaining = target_count - len(anchor_studies)

    # Calculate how many variations per anchor study
    variations_per_study = max(1, remaining // len(anchor_studies))

    with tqdm(total=remaining, desc="Generating AI variations") as pbar:
        for anchor in anchor_studies[:50]:  # Use first 50 as templates
            if len(expanded_studies) >= target_count:
                break

            prompt = f"""Generate {variations_per_study} variations of this business case study.
Each variation should be a DIFFERENT company in a DIFFERENT industry or geography,
but following a similar business model pattern.

ORIGINAL STUDY:
Industry: {anchor['industry']}
Business Model: {anchor['business_model']}
Company: {anchor['company_name']}
Summary: {anchor['summary']}
Revenue Model: {anchor.get('revenue_model', '')}
Pricing Strategy: {anchor.get('pricing_strategy', '')}

Generate {variations_per_study} NEW case studies as JSON array:
[
  {{
    "industry": "<different industry>",
    "business_model": "{anchor['business_model']}",
    "company_name": "<fictional but realistic name>",
    "summary": "<2-3 sentence summary>",
    "revenue_model": "<how they make money>",
    "pricing_strategy": "<pricing approach>",
    "legal_structure": "Delaware C-Corp"
  }},
  ...
]

Only output valid JSON array. Be creative and realistic."""

            try:
                response = await llm.ainvoke(prompt)
                content = (
                    response.content if hasattr(response, "content") else str(response)
                )

                # Parse JSON
                variations = json.loads(content)
                expanded_studies.extend(variations)
                pbar.update(len(variations))

                # Rate limiting
                await asyncio.sleep(1.0)

            except Exception as e:
                logger.warning("Failed to generate variations for %s: %s", anchor["company_name"], e)
                continue

            if len(expanded_studies) >= target_count:
                break

    logger.info("✅ Expanded to %d total studies", len(expanded_studies))
    return expanded_studies[:target_count]


def load_seed_data() -> list[dict[str, Any]]:
    """Load anchor case studies from JSON file."""
    data_file = Path(__file__).parent.parent / "data" / "case_studies_seed.json"

    if not data_file.exists():
        logger.error("Seed data file not found: %s", data_file)
        logger.error("Please create the data/case_studies_seed.json file first.")
        sys.exit(1)

    with open(data_file, "r", encoding="utf-8") as f:
        studies = json.load(f)

    logger.info("Loaded %d anchor case studies from %s", len(studies), data_file)
    return studies


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Seed the Golden Database with proven business models"
    )
    parser.add_argument(
        "--expand",
        action="store_true",
        help="Expand anchor studies to 1,000+ using AI generation",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=1000,
        help="Target number of studies when using --expand (default: 1000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of studies to process per batch (default: 50)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Resume from this index (for recovery)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate embeddings but don't insert into database",
    )

    args = parser.parse_args()

    # Validate environment
    if not settings.gemini_api_key:
        logger.error("GEMINI_API_KEY not set in environment")
        sys.exit(1)

    if not settings.supabase_url or not settings.supabase_key:
        logger.error("SUPABASE_URL and SUPABASE_KEY must be set")
        sys.exit(1)

    # Load seed data
    case_studies = load_seed_data()

    # Expand with AI if requested
    if args.expand:
        case_studies = await expand_with_ai(case_studies, args.target_count)

    # Seed the database
    seeder = GoldenDatabaseSeeder(dry_run=args.dry_run)
    await seeder.seed(
        case_studies,
        batch_size=args.batch_size,
        start_index=args.start_index,
    )

    logger.info("🎉 Golden Database seeding complete!")


if __name__ == "__main__":
    asyncio.run(main())
