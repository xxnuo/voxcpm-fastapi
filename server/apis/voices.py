import json
import logging
import os
from typing import Annotated, List, Optional

from fastapi import APIRouter, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from server.config import Config
from server.utils import normalize_str_to_safe_filename

logger = logging.getLogger("voices")
logger.setLevel(Config.LOG_LEVEL)

OPENAI_VOICES = {
    "alloy": Config.DEFAULT_VOICE_ID,
    "ash": Config.DEFAULT_VOICE_ID,
    "ballad": Config.DEFAULT_VOICE_ID,
    "coral": Config.DEFAULT_VOICE_ID,
    "echo": Config.DEFAULT_VOICE_ID,
    "fable": Config.DEFAULT_VOICE_ID,
    "onyx": Config.DEFAULT_VOICE_ID,
    "nova": Config.DEFAULT_VOICE_ID,
    "sage": Config.DEFAULT_VOICE_ID,
    "shimmer": Config.DEFAULT_VOICE_ID,
    "verse": Config.DEFAULT_VOICE_ID,
}


class VoiceInfo(BaseModel):
    id: str
    audio_path: Optional[str] = None
    text: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    uploaded: Optional[bool] = False


VOICE_LIST: List[VoiceInfo] = []


def build_voice_list():
    """[Internal] 构建语音列表"""
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
    for file in os.listdir(Config.USER_VOICES_DIR):
        if file.endswith(".json"):
            user_voice_info_json: dict = json.load(
                open(os.path.join(Config.USER_VOICES_DIR, file), "r", encoding="utf-8")
            )
            user_voice_info = VoiceInfo.model_validate(user_voice_info_json)
            user_voice_info.uploaded = True
            logger.info(f"Loaded user voice: {user_voice_info.id}")
            VOICE_LIST.append(user_voice_info)


def get_voice_audio_full_path(id: str) -> str:
    """[Internal] 获取音频文件完整路径"""
    if len(VOICE_LIST) == 0:
        build_voice_list()
    if id in OPENAI_VOICES:
        id = OPENAI_VOICES[id]
    selected_voice = next((v for v in VOICE_LIST if v.id == id), None)
    if not selected_voice:
        logger.warning(
            f"Voice '{id}' not found, using default voice {Config.DEFAULT_VOICE_ID}."
        )
        selected_voice = next(
            (v for v in VOICE_LIST if v.id == Config.DEFAULT_VOICE_ID), None
        )
        if not selected_voice:
            raise HTTPException(
                status_code=500,
                detail=f"Default voice {Config.DEFAULT_VOICE_ID} not found.",
            )
    if not selected_voice.uploaded:
        return os.path.join(Config.VOICES_DIR, selected_voice.audio_path)
    else:
        return os.path.join(Config.USER_VOICES_DIR, selected_voice.audio_path)


def get_voice_text(id: str) -> str:
    """[Internal] 获取语音文本"""
    if len(VOICE_LIST) == 0:
        build_voice_list()
    if id in OPENAI_VOICES:
        id = OPENAI_VOICES[id]
    selected_voice = next((v for v in VOICE_LIST if v.id == id), None)
    if not selected_voice:
        logger.warning(
            f"Voice '{id}' not found, using default voice {Config.DEFAULT_VOICE_ID}."
        )
        selected_voice = next(
            (v for v in VOICE_LIST if v.id == Config.DEFAULT_VOICE_ID), None
        )
        if not selected_voice:
            raise HTTPException(
                status_code=500,
                detail=f"Default voice {Config.DEFAULT_VOICE_ID} not found.",
            )
    return selected_voice.text or ""


voices_router = APIRouter(tags=["Voices API"])


@voices_router.get("")
def list_voices():
    """List all available voices' ids."""
    if len(VOICE_LIST) == 0:
        build_voice_list()
    try:
        voice = [voice.id for voice in VOICE_LIST] if VOICE_LIST else []
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
def get_voice_info(voice: str) -> VoiceInfo:
    """[Special] Get the info of a voice."""
    if len(VOICE_LIST) == 0:
        build_voice_list()
    selected_voice = next((v for v in VOICE_LIST if v.id == voice), None)
    if not selected_voice:
        raise HTTPException(status_code=400, detail=f"Voice '{voice}' not found.")
    return selected_voice


@voices_router.get("/{voice}/audio")
def get_voice_audio(voice: str) -> FileResponse:
    """[Special] Get the audio file of a voice."""
    if len(VOICE_LIST) == 0:
        build_voice_list()
    selected_voice = next((v for v in VOICE_LIST if v.id == voice), None)
    if not selected_voice:
        raise HTTPException(status_code=400, detail=f"Voice '{voice}' not found.")
    if not selected_voice.uploaded:
        return FileResponse(os.path.join(Config.VOICES_DIR, selected_voice.audio_path))
    else:
        return FileResponse(
            os.path.join(Config.USER_VOICES_DIR, selected_voice.audio_path)
        )


@voices_router.post("/upload")
def upload_voice(
    audio_file: UploadFile,
    text: Annotated[str, Form(description="The text of the audio file.")],
    id: Annotated[str, Form(description="The id of the voice, must be unique.")],
    name: Annotated[str, Form(description="The name of the voice.")],
    description: Annotated[str, Form(description="The description of the voice.")],
):
    """[Special] Upload a user voice."""
    if len(VOICE_LIST) == 0:
        build_voice_list()
    if not audio_file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="File must be a wav file.")
    id = normalize_str_to_safe_filename(id)
    if id in [v.id for v in VOICE_LIST]:
        raise HTTPException(status_code=400, detail=f"Voice id '{id}' already exists.")
    voice_info = VoiceInfo(
        id=id,
        name=name,
        description=description,
        audio_path=f"{id}.wav",
        text=text,
        uploaded=True,
    )

    file_path = os.path.join(Config.USER_VOICES_DIR, f"{id}.wav")
    try:
        with open(file_path, "wb") as f:
            f.write(audio_file.file.read())
    except Exception as e:
        logger.error(f"Error uploading voice audio: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading voice: {e}")
    json_path = os.path.join(Config.USER_VOICES_DIR, f"{id}.json")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(voice_info.model_dump_json())
    except Exception as e:
        logger.error(f"Error saving voice info: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving voice info: {e}")

    VOICE_LIST.append(voice_info)
    return voice_info


@voices_router.delete("/{voice}")
def delete_user_voice(voice: str):
    """[Special] Delete a user voice."""
    if len(VOICE_LIST) == 0:
        build_voice_list()
    selected_voice = next((v for v in VOICE_LIST if v.id == voice and v.uploaded), None)
    if not selected_voice:
        raise HTTPException(status_code=400, detail=f"User voice '{voice}' not found.")
    try:
        os.remove(os.path.join(Config.USER_VOICES_DIR, selected_voice.audio_path))
        os.remove(os.path.join(Config.USER_VOICES_DIR, f"{selected_voice.id}.json"))
        VOICE_LIST.remove(selected_voice)
    except Exception as e:
        logger.error(f"Error deleting voice: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting voice: {e}")
    return selected_voice
