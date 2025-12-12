import os

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class ConfigModel(BaseModel):
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes", "on")
    CORS_ENABLED: bool = os.getenv("CORS_ENABLED", "true").lower() in (
        "true",
        "1",
        "yes",
        "on",
    )
    CORS_ALLOW_ORIGIN_REGEX: str = os.getenv("ALLOW_ORIGIN_REGEX", r"^https?://.*$")
    VIOCES_DIR: str = os.getenv("VIOCES_DIR", "./voices")


Config = ConfigModel()
