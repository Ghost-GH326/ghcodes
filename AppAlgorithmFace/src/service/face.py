'''人脸检测、特征提取、特征比对'''
import cv2
import uuid
import base64
import time
import logging
import tempfile
import numpy as np
from src.config import Config
from src.service.alive_alg.alive import LiveFaceDetection, check_alive
from src.service.face_alg.detect_inference import DetectInference
from src.service.face_alg.landmark_inference import LandmarkInference
from src.service.face_alg.antispoofing_inference import AntiSpoofingInference

from src.service.face_alg.service_demo import PthExtractor, detect_align, get_sim
from src.service.image import ImageService
from src.util.decorator import catch


class FaceService:
    PREPARE_CHECK = 0
    ALIVE_CHECK = 1

    @classmethod
    def init(cls):
        # 对齐模型
        cls.detector = DetectInference(Config.DETECT_MODEL_PATH, Config.DETECT_GPUID, Config.DETECT_INPUT_SHAPE,
                                       Config.DETECT_THRESHOLD)
        cls.landermark_detector = LandmarkInference(Config.DETECT_LANDMARK_MODEL_PATH, Config.DETECT_GPUID)
        # 特征检测模型
        cls.feature_model = PthExtractor(Config.FEATURE_MODEL_PATH,
                                         Config.FEATURE_MODEL_CONFIG,
                                         Config.FEATURE_GPUID,
                                         batch_size=Config.FEATURE_BATCH_SIZE,
                                         im_shape=Config.FEATURE_INPUT_SHAPE)
        # 活体模型
        cls.alive_model = LiveFaceDetection(Config.ALIVE_MODEL_PATH, Config.ALIVE_THRESHOLD_ORIENTATION,
                                            Config.ALIVE_THRESHOLD_MOUSE, Config.ALIVE_THRESHOLD_MOVE,
                                            Config.ALIVE_THRESHOLD_EYE, Config.ALIVE_THRESHOLD_ALIGH,
                                            Config.ALIVE_WIN_WIDTH, Config.ALIVE_WIN_HEIGHT)
        # 人脸防伪模型
        cls.anti_model = AntiSpoofingInference(model_path=Config.ANTI_MODEL_PATH, use_gpu=Config.ANTI_GPUID != -1)

    @classmethod
    @catch(5)
    async def get_feature_value(cls, base64_str=None, url=None, image=None):
        '''
        base64_str: 图片 base64 字符串
        url: 图片 url
        image: cv2 的 image 对象
        依次判断三个参数，任一个满足时开始进行特征提取
        返回识别的特征值
        '''
        # 处理二进制
        if base64_str is not None:
            binary = base64.b64decode(base64_str)
        elif url is not None:
            binary = await ImageService.download_image(url)
        else:
            binary = None
        # 二进制或 image 对象需要至少存在一个
        if binary is None and image is None:
            return None
        filename = uuid.uuid4().hex
        if binary is not None:
            # buf = np.asarray(bytearray(binary), dtype="uint8")
            buf = np.frombuffer(binary, dtype="uint8")
            image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        logging.debug("detect start: %s", time.time())
        key_imarray_map = detect_align(cls.detector, {filename: image}, align_shape=Config.DETECT_ALIGN_SHAPE)
        logging.debug("detect done: %s", time.time())
        feature_map = cls.feature_model.inference(key_imarray_map)
        logging.debug("feature done: %s", time.time())
        return feature_map[filename]

    @classmethod
    @catch(5)
    async def get_compare_score(cls, feature_1, feature_2):
        '''
        根据特征计算相似度得分
        '''
        return get_sim(feature_1, feature_2)

    @classmethod
    @catch(5)
    async def alive_check(cls, video_b64, action_list, risk_score, **kwargs):
        '''
        传入video base64字符串和动作类型数组
        返回活体检测后的 image 、错误消息、中间数据对象
        '''
        images = []
        with tempfile.NamedTemporaryFile() as f:
            f.write(base64.b64decode(video_b64))
            cap = cv2.VideoCapture(f.name)
            ret = True
            while (cap.isOpened()) and ret:
                ret, frame = cap.read()
                if ret:
                    images.append(frame)
        assert len(images) > 0, "video error, extract zero image"
        return check_alive(cls.alive_model, cls.detector, cls.landermark_detector, images, action_list, cls.ALIVE_CHECK,
                           risk_score, **kwargs)

    @classmethod
    @catch(5)
    async def face_check(cls, image_b64, action_list, **kwargs):
        '''
        传入图像 base64字符串和动作类型数组
        返回活体检测后的 image 、错误消息、中间数据对象
        '''
        binary = base64.b64decode(image_b64)
        # buf = np.asarray(bytearray(binary), dtype="uint8")
        buf = np.frombuffer(binary, dtype="uint8")
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        image, error_msg, error_type, mid_values = check_alive(cls.alive_model, cls.detector, cls.landermark_detector,
                                                               [image], action_list, cls.PREPARE_CHECK, "", **kwargs)
        if error_type in LiveFaceDetection.FACE_PREPARE_OK_ERRORS:
            error_type, error_msg = 0, ""
        return image, error_msg, error_type, mid_values

    @classmethod
    @catch(5)
    async def get_anti_score(cls, url=None, image=None):
        '''
        传入图像 url，获取反欺诈打分
        '''
        if url:
            binary = await ImageService.download_image(url)
            buf = np.frombuffer(binary, dtype="uint8")
            image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return cls.anti_model.process_cv_frame(cls.detector, cls.landermark_detector, image)
