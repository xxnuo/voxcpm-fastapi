import json
import logging
import os
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.config import Config

logger = logging.getLogger("audio")
logger.setLevel(Config.LOG_LEVEL)

audio_router = APIRouter(tags=["Audio API"])


class VoiceInfo(BaseModel):
    name: str
    description: Optional[str] = None
    audio_path: Optional[str] = None
    text_path: Optional[str] = None


VOICE_LIST: List[VoiceInfo] = []


def build_voice_list():
    """Load the voice list from the voices directory"""
    for file in os.listdir(Config.VIOCES_DIR):
        if file.endswith(".json"):
            voice_info_json: dict = json.load(
                open(os.path.join(Config.VIOCES_DIR, file), "r", encoding="utf-8")
            )
            voice_info = VoiceInfo.model_validate(voice_info_json)
            logger.info(f"Loaded voice: {voice_info.name}")
            VOICE_LIST.append(voice_info)


@audio_router.get("/voices")
def list_voices():
    """List all available voices for text-to-speech"""
    if len(VOICE_LIST) == 0:
        build_voice_list()
    try:
        voice = [voice.name for voice in VOICE_LIST]
        return {"voices": voice}
    except Exception as e:
        logger.error(f"Error listing voices: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "server_error",
                "message": "Failed to retrieve voice list",
                "type": "server_error",
            },
        )
