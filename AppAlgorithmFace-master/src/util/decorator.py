'''
装饰器工具集合
'''
import logging
import asyncio
import traceback
import functools
from src.exception import ServerTimeoutError, ServerProcessingError


# 配置超时
def catch(timeout_seconds):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs),
                                              timeout=timeout_seconds)
            except asyncio.TimeoutError:
                logging.error("decorator meet timeout")
                raise ServerTimeoutError()
            except Exception as e:
                logging.error(traceback.format_exc())
                raise ServerProcessingError(msg=str(e))

        return wrapper

    return decorator
