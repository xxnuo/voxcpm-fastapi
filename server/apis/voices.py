import json
import logging
import os
import subprocess
import tempfile
from typing import Annotated, List, Literal, Optional

import soundfile as sf
from fastapi import APIRouter, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from server.config import Config

logger = logging.getLogger("voices")
logger.setLevel(Config.LOG_LEVEL)


class VoiceInfo(BaseModel):
    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    audio_path: Optional[str] = None
    text: Optional[str] = None


VOICE_LIST: List[VoiceInfo] = []


def build_voice_list():
    """构建语音列表"""
    if len(VOICE_LIST) > 0:
        return
    for file in os.listdir(Config.VOICES_DIR):
        if file.endswith(".json"):
            voice_info_json: dict = json.load(
                open(os.path.join(Config.VOICES_DIR, file), "r", encoding="utf-8")
            )
            voice_info = VoiceInfo.model_validate(voice_info_json)
            logger.info(f"Loaded voice: {voice_info.name}")
            VOICE_LIST.append(voice_info)


voices_router = APIRouter(tags=["Voices API"])


@voices_router.get("/")
def list_voices():
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


@voices_router.get("/{voice}")
def get_voice(voice: str):
    if len(VOICE_LIST) == 0:
        build_voice_list()
    selected_voice = next((v for v in VOICE_LIST if v.name == voice), None)
    if not selected_voice:
        raise HTTPException(status_code=400, detail=f"Voice '{voice}' not found.")
    voice_wav_path = os.path.join(Config.VOICES_DIR, selected_voice.audio_path or "")
    if not os.path.exists(voice_wav_path):
        raise HTTPException(
            status_code=400,
            detail=f"Audio file for voice {voice} not found at {voice_wav_path}",
        )
    return FileResponse(voice_wav_path, media_type="audio/wav")


@voices_router.post("/upload")
def upload_voice(
    file: UploadFile,
    text: Annotated[str, Form(description="The text to use for the voice.")],
    description: Annotated[str, Form(description="The description of the voice.")],
):
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="File must be a wav file.")
    file_path = os.path.join(Config.VOICES_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    voice_info = VoiceInfo(
        name=file.filename,
        audio_path=file.filename,
        text_path=None,
        text=None,
    )
    VOICE_LIST.append(voice_info)
    return {"message": "Voice uploaded successfully.", "voice": voice_info.name}
