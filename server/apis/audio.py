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
from voxcpm import VoxCPM
from server.apis.voices import VOICE_LIST, voices_router
from server.apis.models import MODELS

logger = logging.getLogger("audio")
logger.setLevel(Config.LOG_LEVEL)

MODEL: Optional[VoxCPM] = None


def load_model():
    """加载模型"""
    global MODEL
    if MODEL is None:
        logger.info("Loading TTS model...")
        MODEL = VoxCPM.from_pretrained(
            hf_model_id=Config.VOXCPM_MODEL_ID,
            zipenhancer_model_id=Config.ZIPENHANCER_MODEL_ID,
            local_files_only=True,
        )
        logger.info("TTS model loaded.")


SUPPORTED_FORMATS = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
}


audio_router = APIRouter(tags=["Audio API"])
audio_router.include_router(voices_router, prefix="/voices")


class GenerateSpeechRequest(BaseModel):
    # OpenAI Compatible Parameters
    input: str = Field(
        ...,
        description="The text to generate audio for.",
    )
    model: Literal[MODELS.keys()] = Field(
        default="voxcpm-1.5",
        description=f"One of the available TTS models: {', '.join(MODELS.keys())}.",
    )
    voice: Literal[VOICE_LIST.keys()] = Field(
        default="en_female_neko",
        description=f"One of the available voices: {', '.join(VOICE_LIST.keys())}.",
    )
    instructions: Optional[str] = Field(
        default=None,
        description="[Not used] Control the voice of your generated audio with additional instructions. Does not work with tts-1 or tts-1-hd.",
    )
    response_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]] = (
        Field(
            default="mp3",
            description="The format to audio in. Supported formats are mp3, opus, aac, flac, wav, and pcm.",
        )
    )
    speed: Optional[float] = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="[Not used] The speed of the generated audio. Select a value from 0.25 to 4.0. 1.0 is the default.",
    )
    stream_format: Optional[Literal["sse", "audio"]] = Field(
        default="audio",
        description="The format to stream the audio in. Supported formats are sse and audio. sse is not supported for tts-1 or tts-1-hd.",
    )

    # Extra parameters for the audio generation supported by VoxCPM
    prompt_wav_path: Optional[str] = Field(
        default=None,
        description="Path to a prompt speech for voice cloning.(use /voices/upload to upload a voice)",
    )
    prompt_text: Optional[str] = Field(
        default=None,
        description="Reference text for voice cloning.(use /voices/upload to upload a voice)",
    )
    cfg_value: Optional[float] = Field(
        default=2.0,
        description="LM guidance on LocDiT, higher for better adherence to the prompt, but maybe worse.",
    )
    inference_timesteps: Optional[int] = Field(
        default=10,
        description="LocDiT inference timesteps, higher for better result, lower for fast speed.",
    )
    normalize: Optional[bool] = Field(
        default=False,
        description="Enable external TN tool, but will disable native raw text support.",
    )
    denoise: Optional[bool] = Field(
        default=False,
        description="Enable external Denoise tool, but it may cause some distortion and restrict the sampling rate to 16kHz.",
    )
    retry_badcase: Optional[bool] = Field(
        default=True,
        description="Enable retrying mode for some bad cases (unstoppable).",
    )
    retry_badcase_max_times: Optional[int] = Field(
        default=3,
        description="Maximum retrying times.",
    )
    retry_badcase_ratio_threshold: Optional[float] = Field(
        default=6.0,
        description="Maximum length restriction for bad case detection (simple but effective), it could be adjusted for slow pace speech.",
    )


@audio_router.post("/speech")
async def generate_speech(request: GenerateSpeechRequest):
    global MODEL
    if MODEL is None:
        load_model()
    if not request.input or not request.input.strip():
        raise HTTPException(status_code=400, detail="Input is required")

    selected_voice = next((v for v in VOICE_LIST if v.name == request.voice), None)
    if not selected_voice:
        raise HTTPException(
            status_code=400, detail=f"Voice '{request.voice}' not found."
        )

    prompt_audio_path = (
        os.path.join(Config.VOICES_DIR, selected_voice.audio_path)
        if selected_voice.audio_path
        else None
    )
    if not os.path.exists(prompt_audio_path):
        raise HTTPException(
            status_code=400,
            detail=f"Audio file for voice {request.voice} not found at {prompt_audio_path}",
        )
    prompt_text_path = (
        os.path.join(Config.VOICES_DIR, selected_voice.text_path)
        if selected_voice.text_path
        else None
    )
    if not os.path.exists(prompt_text_path):
        raise HTTPException(
            status_code=400,
            detail=f"Text file for voice {request.voice} not found at {prompt_text_path}",
        )
    with open(prompt_text_path, "r", encoding="utf-8") as f:
        prompt_text = f.read()

    try:
        wav = MODEL.generate(
            text=request.input,
            prompt_wav_path=prompt_audio_path,
            prompt_text=prompt_text,
            cfg_value=request.cfg_value or 2.0,
            inference_timesteps=request.inference_timesteps or 10,
            normalize=request.normalize or False,
            denoise=request.denoise or False,
            retry_badcase=request.retry_badcase or True,
            retry_badcase_max_times=request.retry_badcase_max_times or 3,
            retry_badcase_ratio_threshold=request.retry_badcase_ratio_threshold or 6.0,
        )
    except Exception as e:
        logger.error(f"Error generating audio: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating audio: {e}")

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
