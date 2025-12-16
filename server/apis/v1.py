import logging

from fastapi import APIRouter

from server.apis.audio import audio_router
from server.apis.models import models_router
from server.apis.voices import voices_router
from server.config import Config

logger = logging.getLogger("v1")
logger.setLevel(Config.LOG_LEVEL)

router = APIRouter(
    tags=["OpenAI Compatible API for VoxCPM TTS"],
    responses={404: {"description": "Not found"}},
)

@router.get("/health")
def get_health():
    """
    健康检查接口，用于检查服务是否正常
    """
    return {"status": "healthy"}

router.include_router(models_router, prefix="/models")
router.include_router(audio_router, prefix="/audio")
router.include_router(voices_router, prefix="/voices")