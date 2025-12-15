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
    HF_MODEL_ID: str = os.getenv("HF_MODEL_ID", "./models/openbmb/VoxCPM1.5")
    ZIPENHANCER_MODEL_ID: str = os.getenv(
        "ZIPENHANCER_MODEL_ID",
        "./models/iic/speech_zipenhancer_ans_multiloss_16k_base",
    )


Config = ConfigModel()
