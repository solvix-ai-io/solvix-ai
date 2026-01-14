from pydantic import ConfigDict
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8001
    debug: bool = False

    # LLM Provider Selection
    llm_provider: str = "gemini"  # "openai" or "gemini"

    # Gemini Configuration (PRIMARY)
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-3-flash-preview"
    gemini_temperature: float = 0.3
    gemini_max_tokens: int = 2048

    # OpenAI Configuration (FALLBACK)
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-5-nano"
    openai_temperature: float = 0.3
    openai_max_tokens: int = 2000

    # Timeouts
    llm_timeout_seconds: int = 30
    llm_max_retries: int = 3

    # Logging
    log_level: str = "INFO"
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


settings = Settings()
