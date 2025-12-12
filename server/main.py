import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.apis.v1 import router
from server.config import Config
from server.logger import setup_logger

setup_logger()
logger = logging.getLogger("main")
logger.setLevel(Config.LOG_LEVEL)

app = FastAPI(tags=["VoxCPM Server API"])

# Add CORS middleware if enabled
if Config.CORS_ENABLED:
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=Config.CORS_ALLOW_ORIGIN_REGEX,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/health")
def get_health():
    """
    健康检查接口，用于检查服务是否正常
    """
    return {"status": "healthy"}


app.include_router(router, prefix="/v1")
