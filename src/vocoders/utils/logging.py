import sys

from loguru import logger

logger.remove()
logger.add(
    sys.stdout,
    format="<g>{time:MM-DD HH:mm:ss}</g> |<lvl>{level:^8}</lvl>| {file}:{line} | {message}",
    backtrace=True,
    diagnose=True,
)
