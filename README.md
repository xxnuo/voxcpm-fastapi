# voxcpm-fastapi

VoxCPM inference service compatible with OpenAI interface 

兼容 OpenAI 接口的 VoxCPM 推理服务

## Quick Start

```
git clone https://github.com/xxnuo/voxcpm-fastapi.git
cd voxcpm-fastapi
git submodule update --init --recursive
make prepare-model
uv sync
uv pip install -e ./voxcpm
uv run run.py
```

## Usage

Docs: [openai.com/docs/api-reference/audio/createSpeech](https://platform.openai.com/docs/api-reference/audio/createSpeech)
