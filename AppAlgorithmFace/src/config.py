'''
全局配置
'''
import toml
import logging
import socket

_c = toml.load("./conf/config.toml")


class Configer:
    def __init__(self):
        # app
        self.APP = {
            "debug": _c["Server"]["Debug"],
            "gzip": _c["Server"]["Gzip"],
            "autoreload": _c["Server"]["AutoReload"],
        }
        self.PORT = _c["Server"]["Port"]
        self.PROCESS_NUM = _c["Server"]["ProcessNum"]
        self.SUPPORTED_BUSINESS = _c["Server"]["SupportedBusiness"]
        # log
        self.LOG_PATH = _c["Log"]["FilePath"]
        if self.APP["debug"] is True:
            self.LOG_LEVEL = logging.DEBUG
        else:
            self.LOG_LEVEL = logging.INFO
        # local ip
        try:
            self.IP = socket.gethostbyname(socket.getfqdn(socket.gethostname()))
        except Exception:
            self.IP = "127.0.0.1"
        # oss
        self.OSS_ENDPOINT = _c["Oss"]["Endpoint"]
        self.OSS_BUCKET_NAME = _c["Oss"]["BucketName"]
        self.OSS_FOLDER_NAME = _c["Oss"]["FolderName"]
        self.OSS_URL_EXPIRES = _c["Oss"]["UrlExpires"]
        self.OSS_EXECUTORS_PER_PROCESS = _c["Oss"]["ExecutorsPerProcess"]
        self.OSS_ACCESS_KEY_ID = _c["Oss"]["AccessKey"]
        self.OSS_ACCESS_KEY_SECRET = _c["Oss"]["AccessSecret"]
        # detect
        self.DETECT_MODEL_PATH = _c["Detect"]["ModelPath"]
        self.DETECT_LANDMARK_MODEL_PATH = _c["Detect"]["LandmarkModelPath"]
        self.DETECT_INPUT_SHAPE = _c["Detect"]["InputShape"]
        self.DETECT_THRESHOLD = _c["Detect"]["Threshold"]
        self.DETECT_GPUID = _c["Detect"]["GpuID"]
        self.DETECT_ALIGN_SHAPE = _c["Detect"]["AlignShape"]
        # feature
        self.FEATURE_MODEL_PATH = _c["Feature"]["ModelPath"]
        self.FEATURE_MODEL_CONFIG = _c["Feature"]["ModelConfig"]
        self.FEATURE_GPUID = _c["Feature"]["GpuID"]
        self.FEATURE_BATCH_SIZE = _c["Feature"]["BatchSize"]
        self.FEATURE_INPUT_SHAPE = _c["Feature"]["InputShape"]
        self.FEATURE_MATCH_THRESHOLD = _c["Feature"]["MatchThreshold"]
        self.FEATURE_SAVE_DATA_RATE = _c["Feature"]["SaveDataRate"]
        # alive
        self.ALIVE_ACTION_TYPES = _c["Alive"]["ActionTypes"]
        self.ALIVE_SPECIAL_BUSINESS = _c["Alive"]["SpecialBusiness"]
        self.ALIVE_MODEL_PATH = _c["Alive"]["ModelPath"]
        self.ALIVE_THRESHOLD_ORIENTATION = _c["Alive"]["ThresholdOrientation"]
        self.ALIVE_THRESHOLD_MOUSE = _c["Alive"]["ThresholdMouse"]
        self.ALIVE_THRESHOLD_MOVE = _c["Alive"]["ThresholdMove"]
        self.ALIVE_THRESHOLD_EYE = _c["Alive"]["ThresholdEye"]
        self.ALIVE_THRESHOLD_ALIGH = _c["Alive"]["ThresholdAligh"]
        self.ALIVE_WIN_WIDTH = _c["Alive"]["WinWidth"]
        self.ALIVE_WIN_HEIGHT = _c["Alive"]["WinHeight"]
        self.ALIVE_SAVE_DATA_RATE = _c["Alive"]["SaveDataRate"]
        # anti
        self.ANTI_MODEL_PATH = _c["Anti"]["ModelPath"]
        self.ANTI_GPUID = _c["Anti"]["GpuID"]
        self.ANTI_FAKE_THRESHOLD = _c["Anti"]["FakeThreshold"]
        # redis
        self.REDIS_AUTH_URL = _c["Redis"]["AuthUrl"]
        self.REDIS_AUTH_TOKEN = _c["Redis"]["AuthToken"]
        self.REDIS_AUTH_APP = _c["Redis"]["AuthApp"]
        self.REDIS_NAME = _c["Redis"]["Name"]
        self.REDIS_DB = _c["Redis"]["DB"]
        self.REDIS_MIN_SIZE = _c["Redis"]["MinSize"]
        self.REDIS_MAX_SIZE = _c["Redis"]["MaxSize"]
        # session
        self.SESSION_EXPIRE_SECONDS = _c["Session"]["ExpireSeconds"]
        self.SESSION_ALIVE_MAX_USED = _c["Session"]["AliveMaxUsed"]
        self.SESSION_FACE_MAX_USED = _c["Session"]["FaceMaxUsed"]


Config = Configer()
