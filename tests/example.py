import soundfile as sf
import numpy as np
from voxcpm import VoxCPM

model = VoxCPM.from_pretrained(
    hf_model_id="./models/openbmb/VoxCPM1.5",
    zipenhancer_model_id="./models/iic/speech_zipenhancer_ans_multiloss_16k_base",
    local_files_only=True
)

# Non-streaming
wav = model.generate(
    text="VoxCPM is an innovative end-to-end TTS model from ModelBest, designed to generate highly expressive speech.",
    prompt_wav_path=None,  # optional: path to a prompt speech for voice cloning
    prompt_text=None,  # optional: reference text
    cfg_value=2.0,  # LM guidance on LocDiT, higher for better adherence to the prompt, but maybe worse
    inference_timesteps=10,  # LocDiT inference timesteps, higher for better result, lower for fast speed
    normalize=False,  # enable external TN tool, but will disable native raw text support
    denoise=False,  # enable external Denoise tool, but it may cause some distortion and restrict the sampling rate to 16kHz
    retry_badcase=True,  # enable retrying mode for some bad cases (unstoppable)
    retry_badcase_max_times=3,  # maximum retrying times
    retry_badcase_ratio_threshold=6.0,  # maximum length restriction for bad case detection (simple but effective), it could be adjusted for slow pace speech
)

sf.write("output.wav", wav, model.tts_model.sample_rate)
print("saved: output.wav")

# Streaming
chunks = []
for chunk in model.generate_streaming(
    text="Streaming text to speech is easy with VoxCPM!",
    # supports same args as above
):
    chunks.append(chunk)
wav = np.concatenate(chunks)

sf.write("output_streaming.wav", wav, model.tts_model.sample_rate)
print("saved: output_streaming.wav")
