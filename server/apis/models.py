import logging

from fastapi import APIRouter, HTTPException

from server.config import Config

logger = logging.getLogger("models")
logger.setLevel(Config.LOG_LEVEL)

MODELS = {
    # OpenAI Compatible Models
    "tts-1": {
        "id": "tts-1",
        "object": "model",
        "created": 1677649963,
        "owned_by": "openbmb",
    },
    "tts-1-hd": {
        "id": "tts-1-hd",
        "object": "model",
        "created": 1677649963,
        "owned_by": "openbmb",
    },
    "gpt-4o-mini-tts": {
        "id": "gpt-4o-mini-tts",
        "object": "model",
        "created": 1677649963,
        "owned_by": "openbmb",
    },
    # VoxCPM Models
    "voxcpm-1.5": {
        "id": "voxcpm-1.5",
        "object": "model",
        "created": 1677649963,
        "owned_by": "openbmb",
    },
}


models_router = APIRouter(tags=["Models API"])


@models_router.get("")
async def list_models():
    """List all available models"""
    try:
        return {"object": "list", "data": [model for model in MODELS.values()]}
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "server_error",
                "message": "Failed to retrieve model list",
                "type": "server_error",
            },
        )


@models_router.get("/{model}")
async def retrieve_model(model: str):
    """Retrieve a specific model"""
    try:
        # Define available models

        # Check if requested model exists
        if model not in MODELS:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "model_not_found",
                    "message": f"Model '{model}' not found",
                    "type": "invalid_request_error",
                },
            )

        # Return the specific model
        return MODELS[model]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving model {model}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "server_error",
                "message": "Failed to retrieve model information",
                "type": "server_error",
            },
        )
