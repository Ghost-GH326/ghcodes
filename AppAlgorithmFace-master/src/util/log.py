'''
日志工具
'''
import logging
from src.config import Config

logging.basicConfig(
    format=("%(asctime)s %(levelname).4s %(filename)s %(lineno)d [" + Config.IP + "]: %(message)s"),
    datefmt="%Y-%m-%d %H:%M:%S",
    level=Config.LOG_LEVEL,
    filename=Config.LOG_PATH,
)


def log_request(handler):
    if handler.get_status() < 400:
        log_method = logging.info
    elif handler.get_status() < 500:
        log_method = logging.warning
    else:
        log_method = logging.error
    request_time = 1000.0 * handler.request.request_time()
    log_method("%d %s %.2fms", handler.get_status(), handler._request_summary(), request_time)
