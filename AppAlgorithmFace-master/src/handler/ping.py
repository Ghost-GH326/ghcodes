'''
ping handler，用于发布系统点火测试
'''
import os
from src.handler.basic import BasicHandler


class PingHandler(BasicHandler):
    async def get(self):
        data = {
            "pid": os.getpid(),
        }
        return self.write_response(msg="pong!", data=data)

    async def head(self):
        return self.set_status(204)
