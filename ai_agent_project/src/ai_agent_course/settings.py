from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración validada de la aplicación."""

    model_config = SettingsConfigDict(
        extra="ignore",
        case_sensitive=False,
        env_file_encoding="utf-8",
    )

    app_env: Literal["development", "testing", "production"] = "development"
    gemini_api_key: SecretStr | None = None
    use_real_gemini: bool = False
    default_model: str = "gemini-3.1-flash-lite"

    request_timeout_seconds: float = Field(default=20.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=8)
    retry_base_delay_seconds: float = Field(default=0.25, ge=0)
    max_concurrency: int = Field(default=3, ge=1, le=32)
    history_max_turns: int = Field(default=6, ge=1, le=50)

    input_usd_per_million: float = Field(default=0.25, ge=0)
    output_usd_per_million: float = Field(default=1.50, ge=0)

    @model_validator(mode="after")
    def validate_live_mode(self):
        if self.use_real_gemini and self.gemini_api_key is None:
            raise ValueError("USE_REAL_GEMINI=1 requiere GEMINI_API_KEY")
        return self

    @classmethod
    def from_env(cls, env_path: str | Path | None = None):
        path = Path(env_path) if env_path is not None else None
        values = {"_env_file": path} if path is not None and path.exists() else {}
        return cls(**values)

    def api_key_value(self) -> str | None:
        return self.gemini_api_key.get_secret_value() if self.gemini_api_key else None

    def safe_dict(self) -> dict:
        data = self.model_dump(exclude={"gemini_api_key"})
        data["gemini_api_key"] = "***configurada***" if self.gemini_api_key else None
        return data
