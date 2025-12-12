import logging

from server.config import Config


def setup_logger():
    logging.basicConfig(level=Config.LOG_LEVEL)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("python_multipart").setLevel(logging.WARNING)
