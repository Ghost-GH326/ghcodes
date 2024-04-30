'''
handler 基类
'''
import logging
import tornado.web
from tornado.escape import json_decode
from src.exception import BaseResponse, IllegalJsonBody


class BasicHandler(tornado.web.RequestHandler):
    def __init__(self, *args, **kwargs):
        tornado.web.RequestHandler.__init__(self, *args, **kwargs)

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "X-Requested-With, Content-Type")
        self.set_header("Access-Control-Allow-Methods", "POST, GET, PUT, OPTIONS, DELETE, PATCH")

    def options(self):
        pass

    def get_query_args(self):
        # 获取所有 url 参数，重复的会覆盖最后一个
        result = {}
        for k, v in self.request.query_arguments.items():
            result[k] = v[-1].decode()
        return result

    def get_form_args(self):
        # 获取所有 form 参数，重复的会覆盖最后一个
        result = {}
        for k, v in self.request.arguments.items():
            result[k] = v[-1].decode()
        return result

    def get_json_args(self):
        try:
            return json_decode(self.request.body)
        except (ValueError, TypeError):
            raise IllegalJsonBody()

    def write_response(self, resp=BaseResponse, msg="", data=dict()):
        '''统一返回值
        格式为：
        {
            "code": 0 或其他错误码
            "msg": "ok" 或返回值说明
            "data": 要返回的数据字典
        }
        '''
        if resp.error_code != 0:
            logging.warn(msg or resp.msg)
        # 全部返回 200，调用方根据 error_code 判断业务逻辑
        # self.set_status(resp.http_code)
        self.write({
            "code": resp.error_code,
            "msg": msg or resp.msg,
            "data": data,
        })
