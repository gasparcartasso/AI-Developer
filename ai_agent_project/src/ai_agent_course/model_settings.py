from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelSettings(BaseSettings):
    """Configuración de modelos cloud y locales para la Clase 4."""

    model_config = SettingsConfigDict(
        extra="ignore",
        case_sensitive=False,
        env_file_encoding="utf-8",
    )

    gemini_model: str = "gemini-3.1-flash-lite"
    ollama_model: str = "qwen3.5:4b"
    ollama_base_url: str = "http://localhost:11434"
    timeout_seconds: float = Field(default=30.0, gt=0)

    use_real_gemini: bool = False
    use_real_ollama: bool = False
    gemini_api_key: SecretStr | None = None

    @model_validator(mode="after")
    def validate_cloud(self):
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
