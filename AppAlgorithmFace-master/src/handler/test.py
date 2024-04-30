'''
face handler，人脸识别接口
'''
import logging
import uuid
from voluptuous import Schema, REMOVE_EXTRA, Required, All, Length
from src.handler.basic import BasicHandler
from src.exception import ExtractFeatureError, IllegalJsonBody, BaseResponse, IllegalParameter
from src.service.face import FaceService
from src.service.oss import OssService
from src.config import Config


class FaceFeatureHandler(BasicHandler):
    schema = Schema({Required("image"): All(str, Length(min=1))}, extra=REMOVE_EXTRA)

    async def post(self):
        '''
        传入图片 base64 字符串，返回特征值
        '''
        # 校验参数
        try:
            q = self.get_json_args()
            params = self.schema(q)
        except Exception as e:
            return self.write_response(resp=IllegalJsonBody, msg=str(e))
        # 执行算法逻辑
        try:
            feature = await FaceService.get_feature_value(params["image"])
        except BaseResponse as e:
            return self.write_response(resp=e)
        if not feature:
            return self.write_response(resp=ExtractFeatureError)
        logging.debug("feature len: %d", len(feature))
        return self.write_response(data={"feature": feature})


class UploadVideoHandler(BasicHandler):
    async def post(self):
        try:
            video = self.request.files.get("video")
            assert video is not None and len(video) == 1, "need one video"
        except Exception as e:
            return self.write_response(resp=IllegalParameter, msg=str(e))
        key = "{}/video/{}".format(Config.OSS_FOLDER_NAME, uuid.uuid4().hex)
        url = await OssService.upload(key, video[0]["body"], video[0]["content_type"])
        return self.write_response(data={"url": url})
