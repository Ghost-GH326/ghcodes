'''
自定义 exception 集合
'''


class BaseResponse(Exception):
    http_code = 200
    error_code = 0
    msg = 'ok'

    def __init__(self, msg=None):
        super().__init__()
        if msg is not None:
            self.msg = msg

    def __str__(self):
        return self.msg


class IllegalParameter(BaseResponse):
    http_code = 200
    error_code = 40001
    msg = 'Illegal request parameters.'


class IllegalJsonBody(BaseResponse):
    http_code = 200
    error_code = 40002
    msg = 'Illegal request json parameters.'


class IllegalSession(BaseResponse):
    http_code = 200
    error_code = 40003
    msg = 'Illegal request session.'


class IllegalSignup(BaseResponse):
    http_code = 200
    error_code = 40006
    msg = 'User already exists.'


class UserNotExists(BaseResponse):
    http_code = 200
    error_code = 40007
    msg = 'User not exists.'


class ServerProcessingError(BaseResponse):
    http_code = 200
    error_code = 50001
    msg = "server processing error"


class ServerTimeoutError(BaseResponse):
    http_code = 200
    error_code = 50002
    msg = "algorithm processing timeout"


class ModelNotAvailableError(BaseResponse):
    http_code = 200
    error_code = 50003
    msg = "algorithm model not ready, wait for one minute please"


class ExtractFeatureError(BaseResponse):
    http_code = 200
    error_code = 50004
    msg = "extract feature error"
