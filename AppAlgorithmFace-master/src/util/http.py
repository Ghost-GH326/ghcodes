'''
http 工具
'''
import aiohttp

session = None


def init():
    global session
    timeout = aiohttp.ClientTimeout(total=3)
    session = aiohttp.ClientSession(timeout=timeout)
