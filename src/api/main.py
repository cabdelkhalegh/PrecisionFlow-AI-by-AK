"""Entry point for the PrecisionFlow API server."""

from __future__ import annotations

import uvicorn

from src.core.config import settings


def main() -> None:
    """Start the Uvicorn server."""
    uvicorn.run(
        "src.api.app:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )


if __name__ == "__main__":
    main()
