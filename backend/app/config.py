from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Database
    database_url: str = "sqlite+aiosqlite:///./ct_seg.db"

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # Storage
    upload_dir: Path = Path("uploads")
    weights_dir: Path = Path("weights")

    # CT windowing defaults
    ct_window_center: int = 50
    ct_window_width: int = 400

    # Model settings
    img_size: int = 256
    num_classes: int = 3
    max_cached_models: int = 2

    # CORS
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:80", "http://frontend"]

    def model_post_init(self, __context) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
