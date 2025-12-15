import json
import logging
import os
import subprocess
import tempfile
from typing import List, Optional

import soundfile as sf
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from server.apis.models import MODELS
from server.config import Config
from voxcpm import VoxCPM

logger = logging.getLogger("audio")
logger.setLevel(Config.LOG_LEVEL)


class VoiceInfo(BaseModel):
    name: str
    description: Optional[str] = None
    audio_path: Optional[str] = None
    text_path: Optional[str] = None
    text: Optional[str] = None


VOICE_LIST: List[VoiceInfo] = []
MODEL: Optional[VoxCPM] = None
SUPPORTED_FORMATS = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
}


def load_model():
    global MODEL
    if MODEL is None:
        logger.info("Loading TTS model...")
        MODEL = VoxCPM.from_pretrained(
            hf_model_id=Config.HF_MODEL_ID,
            zipenhancer_model_id=Config.ZIPENHANCER_MODEL_ID,
            local_files_only=True,
        )
        logger.info("TTS model loaded.")


def build_voice_list():
    if len(VOICE_LIST) > 0:
        return
    for file in os.listdir(Config.VIOCES_DIR):
        if file.endswith(".json"):
            voice_info_json: dict = json.load(
                open(os.path.join(Config.VIOCES_DIR, file), "r", encoding="utf-8")
            )
            voice_info = VoiceInfo.model_validate(voice_info_json)
            logger.info(f"Loaded voice: {voice_info.name}")
            VOICE_LIST.append(voice_info)


build_voice_list()
load_model()

audio_router = APIRouter(tags=["Audio API"])


@audio_router.get("/voices")
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


class CreateSpeechRequest(BaseModel):
    input: str
    model: str
    voice: str
    response_format: Optional[str] = "mp3"
    speed: Optional[float] = 1.0


@audio_router.post("/speech")
async def generate_speech(request: CreateSpeechRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    if not request.input or not request.input.strip():
        raise HTTPException(status_code=400, detail="Input is required")
    speed = request.speed if request.speed is not None else 1.0
    if speed < 0.25 or speed > 4.0:
        raise HTTPException(
            status_code=400, detail="Speed must be between 0.25 and 4.0"
        )

    voice_info = next((v for v in VOICE_LIST if v.name == request.voice), None)
    if not voice_info:
        raise HTTPException(
            status_code=400, detail=f"Voice '{request.voice}' not found."
        )

    prompt_audio_path = (
        os.path.join(Config.VIOCES_DIR, voice_info.audio_path)
        if voice_info.audio_path
        else None
    )
    if not os.path.exists(prompt_audio_path):
        raise HTTPException(
            status_code=400,
            detail=f"Audio file for voice {request.voice} not found at {prompt_audio_path}",
        )
    prompt_text_path = (
        os.path.join(Config.VIOCES_DIR, voice_info.text_path)
        if voice_info.text_path
        else None
    )
    if not os.path.exists(prompt_text_path):
        raise HTTPException(
            status_code=400,
            detail=f"Text file for voice {request.voice} not found at {prompt_text_path}",
        )
    with open(prompt_text_path, "r", encoding="utf-8") as f:
        prompt_text = f.read()

    wav = MODEL.generate(
        text=request.input,
        prompt_wav_path=prompt_audio_path,
        prompt_text=prompt_text,
        cfg_value=2.0,
        inference_timesteps=10,
        normalize=False,
        denoise=False,
        retry_badcase=True,
        retry_badcase_max_times=3,
        retry_badcase_ratio_threshold=6.0,
    )

    response_format = (request.response_format or "mp3").lower()
    if response_format not in SUPPORTED_FORMATS:
        response_format = "mp3"
    media_type = SUPPORTED_FORMATS[response_format]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav_file:
        sf.write(tmp_wav_file.name, wav, MODEL.tts_model.sample_rate)
        input_path = tmp_wav_file.name

    if response_format == "wav":
        output_path = input_path
    else:
        output_path = f"{input_path}.{response_format}"
        try:
            subprocess.run(
                ["ffmpeg", "-i", input_path, "-y", output_path],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg conversion failed: {e.stderr}")
            raise HTTPException(status_code=500, detail="Audio conversion failed.")
        except FileNotFoundError:
            logger.error(
                "ffmpeg not found. Please install ffmpeg to support audio conversion."
            )
            raise HTTPException(
                status_code=500,
                detail="ffmpeg not found, cannot convert audio.",
            )
        finally:
            if input_path != output_path:
                os.remove(input_path)

    def file_iterator(file_path):
        with open(file_path, "rb") as f:
            yield from f
        os.remove(file_path)

    headers = {
        "Content-Disposition": f'attachment; filename="speech.{response_format}"'
    }
    return StreamingResponse(
        file_iterator(output_path), media_type=media_type, headers=headers
    )
