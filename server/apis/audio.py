import json
import logging
import os
import subprocess
import tempfile
from typing import Annotated, List, Literal, Optional

import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from server.config import Config
from voxcpm import VoxCPM

logger = logging.getLogger("audio")
logger.setLevel(Config.LOG_LEVEL)


def load_model():
    global MODEL
    if MODEL is None:
        logger.info("Loading TTS model...")
        MODEL = VoxCPM.from_pretrained(
            hf_model_id=Config.VOXCPM_MODEL_ID,
            zipenhancer_model_id=Config.ZIPENHANCER_MODEL_ID,
            local_files_only=True,
        )
        logger.info("TTS model loaded.")


def build_voice_list():
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


@audio_router.get("/voices/{voice}")
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


@audio_router.post("/voices/upload")
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


class GenerateSpeechRequest(BaseModel):
    # OpenAI Compatible Parameters
    input: str = Field(
        ...,
        description="The text to generate audio for.",
    )
    model: str = Field(
        default="voxcpm-1.5",
        description="One of the available TTS models: tts-1, tts-1-hd or gpt-4o-mini-tts.",
    )
    voice: str = Field(
        default="en_female_neko",
        description="The voice to use for the audio generation.",
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
        description="Path to a prompt speech for voice cloning.",
    )
    prompt_text: Optional[str] = Field(
        default=None,
        description="Reference text for voice cloning.",
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
