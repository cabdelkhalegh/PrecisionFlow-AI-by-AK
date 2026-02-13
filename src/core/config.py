"""Application configuration loaded from environment variables."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """All runtime settings.  Loaded from ``.env`` or environment variables."""

    # --- LLM (Gemini) ---
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    llm_model: str = Field(default="gemini-2.0-flash", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.1, alias="LLM_TEMPERATURE")

    # --- Supabase ---
    supabase_url: str = Field(default="", alias="SUPABASE_URL")
    supabase_key: str = Field(default="", alias="SUPABASE_KEY")

    # --- External APIs ---
    google_trends_api_key: str = Field(default="", alias="GOOGLE_TRENDS_API_KEY")
    semrush_api_key: str = Field(default="", alias="SEMRUSH_API_KEY")
    reddit_client_id: str = Field(default="", alias="REDDIT_CLIENT_ID")
    reddit_client_secret: str = Field(default="", alias="REDDIT_CLIENT_SECRET")
    google_maps_api_key: str = Field(default="", alias="GOOGLE_MAPS_API_KEY")

    # --- Quality Gates ---
    critic_pass_threshold: float = Field(
        default=0.80,
        alias="CRITIC_PASS_THRESHOLD",
        description="Minimum critic score (0-1) required to pass the quality gate.",
    )
    max_critic_retries: int = Field(
        default=3,
        alias="MAX_CRITIC_RETRIES",
        description="Maximum times the Generator can re-draft before escalating.",
    )

    # --- Server ---
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
