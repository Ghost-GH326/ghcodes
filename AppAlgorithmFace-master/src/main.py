'''
主程序，启动程序
'''
import logging
import asyncio
import tornado.web
from src.url import url_patterns
from src.config import Config
from src.util.log import log_request
from src.util import http, redis
from src.service import init_services


class TornadoApplication(tornado.web.Application):
    def __init__(self):
        tornado.web.Application.__init__(self, url_patterns, log_function=log_request, **Config.APP)


def main():
    logging.getLogger().setLevel(Config.LOG_LEVEL)

    app = TornadoApplication()
    server = tornado.httpserver.HTTPServer(app)
    server.bind(Config.PORT)
    server.start(Config.PROCESS_NUM)

    loop = asyncio.get_event_loop()
    # 初始化工具
    http.init()
    redis.init()
    # 初始化服务
    init_services()
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
