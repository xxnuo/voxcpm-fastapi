FROM vllm:r36.4.tegra-aarch64-cu126-22.04-cudastack_standard

# Install OpenBLAS
RUN --mount=type=cache,target=/var/lib/apt/lists \
    apt update && \
    apt install -y --no-install-recommends \
    libopenblas-dev \
    libomp-dev

# Install Python Headers
RUN --mount=type=cache,target=/var/lib/apt/lists \
    apt update && \
    apt install -y --no-install-recommends \
    python3-dev

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.9.16 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_SYSTEM_PYTHON=1

# Install ffmpeg
RUN --mount=type=bind,source=./models/ffmpeg,target=/app/models/ffmpeg \
    cp -r /app/models/ffmpeg/dist/* /usr/local/ && \
    ldconfig

# Copy models
COPY models/openbmb/VoxCPM1.5 /app/models/openbmb/VoxCPM1.5
COPY models/iic/speech_zipenhancer_ans_multiloss_16k_base /app/models/iic/speech_zipenhancer_ans_multiloss_16k_base
COPY models/iic/SenseVoiceSmall /app/models/iic/SenseVoiceSmall
COPY voxcpm /app/voxcpm
COPY run.py /app/run.py

ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VOXCPM=1.5.0
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -e /app/voxcpm -i http://wa.lan:10608/simple --trusted-host wa.lan

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=./pyproject.toml,target=/pyproject.toml \
    uv pip install -r /pyproject.toml -i http://wa.lan:10608/simple --trusted-host wa.lan

COPY dist/server /app/server

WORKDIR /app
EXPOSE 3000
ENV HF_HUB_OFFLINE=1
ENV HF_MODEL_ID=/app/models/openbmb/VoxCPM1.5
ENV ZIPENHANCER_MODEL_ID=/app/models/iic/speech_zipenhancer_ans_multiloss_16k_base

CMD python3 run.py
