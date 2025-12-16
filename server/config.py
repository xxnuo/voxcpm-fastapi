import os

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class ConfigModel(BaseModel):
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes", "on")

    ROOT_DIR: str = os.getenv(
        "ROOT_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    CORS_ENABLED: bool = os.getenv("CORS_ENABLED", "true").lower() in (
        "true",
        "1",
        "yes",
        "on",
    )
    CORS_ALLOW_ORIGIN_REGEX: str = os.getenv("ALLOW_ORIGIN_REGEX", r"^https?://.*$")

    # Model
    VOXCPM_MODEL_ID: str = os.getenv(
        "HF_MODEL_ID",
        os.path.join(ROOT_DIR, "models/openbmb/VoxCPM1.5"),
    )
    ZIPENHANCER_MODEL_ID: str = os.getenv(
        "ZIPENHANCER_MODEL_ID",
        os.path.join(ROOT_DIR, "models/iic/speech_zipenhancer_ans_multiloss_16k_base"),
    )
    ASR_MODEL_ID: str = os.getenv(
        "ASR_MODEL_ID",
        os.path.join(ROOT_DIR, "models/iic/SenseVoiceSmall"),
    )

    # Voices
    VOICES_DIR: str = os.getenv("VOICES_DIR", os.path.join(ROOT_DIR, "voices"))
    USER_VOICES_DIR: str = os.getenv(
        "USER_VOICES_DIR", os.path.join(ROOT_DIR, "user_voices")
    )

    # Default values
    DEFAULT_VOICE_ID: str = os.getenv("DEFAULT_VOICE_ID", "default")
    DEFAULT_CFG_VALUE: float = float(os.getenv("DEFAULT_CFG_VALUE", 2.0))
    DEFAULT_INFERENCE_TIMESTEPS: int = int(os.getenv("DEFAULT_INFERENCE_TIMESTEPS", 10))
    DEFAULT_NORMALIZE: bool = os.getenv("DEFAULT_NORMALIZE", "false").lower() in (
        "true",
        "1",
        "yes",
        "on",
    )
    DEFAULT_DENOISE: bool = os.getenv("DEFAULT_DENOISE", "false").lower() in (
        "true",
        "1",
        "yes",
        "on",
    )
    DEFAULT_RETRY_BADCASE: bool = os.getenv(
        "DEFAULT_RETRY_BADCASE", "true"
    ).lower() in ("true", "1", "yes", "on")
    DEFAULT_RETRY_BADCASE_MAX_TIMES: int = int(
        os.getenv("DEFAULT_RETRY_BADCASE_MAX_TIMES", 3)
    )
    DEFAULT_RETRY_BADCASE_RATIO_THRESHOLD: float = float(
        os.getenv("DEFAULT_RETRY_BADCASE_RATIO_THRESHOLD", 6.0)
    )


Config = ConfigModel()

# Init
os.makedirs(Config.USER_VOICES_DIR, exist_ok=True)
