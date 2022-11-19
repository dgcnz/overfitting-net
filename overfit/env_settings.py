from pathlib import Path

from pydantic import BaseSettings

env_dir = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    MLFLOW_TRACKING_URI: str = ""

    class Config:
        env_file = f"{env_dir}/.env"


settings = Settings()
