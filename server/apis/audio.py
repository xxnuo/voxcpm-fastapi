import base64
import json
import logging
import os
import tempfile
from typing import Literal, Optional

import soundfile as sf
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pydub import AudioSegment

from server.apis.voices import (
    get_voice_audio_full_path,
    get_voice_text,
)
from server.config import Config
from voxcpm import VoxCPM

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
    # OpenAI Compatible Formats
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
}

audio_router = APIRouter(tags=["Audio API"])


class GenerateSpeechRequest(BaseModel):
    # OpenAI Compatible Parameters
    input: str = Field(
        default="What is the weather like today?",
        description="The text to generate audio for.",
    )
    model: str = Field(
        default="voxcpm-1.5",
        description="One of the available TTS models (use /models to list all available models).",
    )
    voice: str = Field(
        default=Config.DEFAULT_VOICE_ID,
        description="One of the available voices (use /voices to list all available voices' ids, use /voices/upload to upload a user voice and get the id).",
    )
    instructions: Optional[str] = Field(
        default=None,
        description="[Not used] Control the voice of your generated audio with additional instructions.",
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

    prompt_audio_path = get_voice_audio_full_path(request.voice)
    prompt_text = get_voice_text(request.voice)

    if not os.path.exists(prompt_audio_path) or not prompt_text:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt audio or text of voice id:{request.voice} not found.",
        )

    try:
        wav = MODEL.generate(
            text=request.input,
            prompt_wav_path=prompt_audio_path,
            prompt_text=prompt_text,
            cfg_value=request.cfg_value or Config.DEFAULT_CFG_VALUE,
            inference_timesteps=request.inference_timesteps
            or Config.DEFAULT_INFERENCE_TIMESTEPS,
            normalize=request.normalize or Config.DEFAULT_NORMALIZE,
            denoise=request.denoise or Config.DEFAULT_DENOISE,
            retry_badcase=request.retry_badcase or Config.DEFAULT_RETRY_BADCASE,
            retry_badcase_max_times=request.retry_badcase_max_times
            or Config.DEFAULT_RETRY_BADCASE_MAX_TIMES,
            retry_badcase_ratio_threshold=request.retry_badcase_ratio_threshold
            or Config.DEFAULT_RETRY_BADCASE_RATIO_THRESHOLD,
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
        try:
            song = AudioSegment.from_wav(input_path)
            output_path = f"{input_path}.{response_format}"
            song.export(output_path, format=response_format)
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            raise HTTPException(status_code=500, detail=f"Audio conversion failed: {e}")
        finally:
            if input_path != output_path:
                os.remove(input_path)

    def file_iterator(file_path):
        with open(file_path, "rb") as f:
            yield from f
        os.remove(file_path)

    def sse_iterator(file_path):
        try:
            with open(file_path, "rb") as f:
                while True:
                    chunk = f.read(4096)
                    if not chunk:
                        break
                    audio_b64 = base64.b64encode(chunk).decode("ascii")
                    data = {
                        "type": "response.audio.delta",
                        "audio": audio_b64,
                    }
                    yield f"data: {json.dumps(data)}\n\n"
            yield f"data: {json.dumps({'type': 'response.audio.completed'})}\n\n"
            yield f"data: {json.dumps({'type': 'response.completed'})}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            os.remove(file_path)

    if request.stream_format == "sse":
        headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked",
        }
        return StreamingResponse(
            sse_iterator(output_path),
            media_type="text/event-stream",
            headers=headers,
        )

    headers = {
        "Content-Disposition": f'attachment; filename="speech.{response_format}"'
    }
    return StreamingResponse(
        file_iterator(output_path), media_type=media_type, headers=headers
    )
