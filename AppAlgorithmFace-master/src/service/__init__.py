from .face import FaceService
from .oss import OssService


def init_services():
    FaceService.init()
    OssService.init()
