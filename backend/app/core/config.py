from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "PD2 Backend"
    api_prefix: str = "/api"
    environment: str = "dev"
    log_level: str = "INFO"

    # Futuro: ruta del modelo, bucket, etc.
    model_path: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

settings = Settings()