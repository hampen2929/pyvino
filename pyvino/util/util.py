import os
import urllib.request
import logging


def get_logger():
    logger = logging.getLogger("download_models")
    logger.setLevel(20)
    sh = logging.StreamHandler()
    logger.addHandler(sh)
    return logger


