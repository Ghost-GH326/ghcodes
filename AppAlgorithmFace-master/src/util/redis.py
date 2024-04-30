'''
redis 工具
'''
import asyncio
import aioredis
from src.config import Config
from src.util import http

redis_pool = None
FACE_USED = "face_used"
ALIVE_USED = "alive_used"
HISTORY = "history"


def init():
    global redis_pool

    async def init_redis_pool():
        async with http.session.post(Config.REDIS_AUTH_URL,
                                     headers={
                                         "REDIS_PROJECT_APPID": Config.REDIS_AUTH_APP,
                                         "REDIS_PROJECT_TOKEN": Config.REDIS_AUTH_TOKEN
                                     },
                                     json={"redisList": [Config.REDIS_NAME]}) as resp:
            assert resp.status == 200, await resp.text()
            meta = await resp.json()
        assert meta["code"] == 0, meta["msg"]
        master = meta["data"][0]["master"][0]
        host, port = master.split(":")
        return await aioredis.create_redis_pool((host, port),
                                                db=Config.REDIS_DB,
                                                password=meta["data"][0]["password"],
                                                minsize=Config.REDIS_MIN_SIZE,
                                                maxsize=Config.REDIS_MAX_SIZE,
                                                encoding="utf-8")

    redis_pool = asyncio.get_event_loop().run_until_complete(init_redis_pool())
