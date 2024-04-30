'''
face handler,人脸识别接口
'''
import io
import json
import asyncio
import time
import uuid
import base64
import random
import cv2
import logging
import traceback
from src.handler.basic import BasicHandler
from voluptuous import Schema, REMOVE_EXTRA, Required, All, Length, In, Optional, Url
from src.exception import IllegalJsonBody, IllegalParameter, IllegalSession, BaseResponse, ServerProcessingError
from src.service.face import FaceService
from src.service.oss import OssService
from src.service.alive_alg.alive import LiveFaceDetection
from src.config import Config
from src.util import redis


class FaceAlivePrepareHandler(BasicHandler):
    schema = Schema(
        {
            Required("actions"): All([int], Length(min=1, max=1)),  # 1=眨眼，2=张嘴, 3=摇头
            Required("sessionId"): All(str, Length(min=1)),
            Required("resource"): All(str, Length(min=1)),
            Required("sign"): All(str, Length(min=1)),
            Required("business"): All(str, In(Config.SUPPORTED_BUSINESS)),
        },
        extra=REMOVE_EXTRA)

    async def post(self):
        '''
        传入图片、动作类型数组、sessionId
        判断人脸是否正常
        sign: resource+sessionId拼接后的 md5 签名
        '''
        # 校验参数
        try:
            q = self.get_json_args()
            params = self.schema(q)
            for i in params["actions"]:
                assert i in Config.ALIVE_ACTION_TYPES, "illegal action {}".format(i)
        except Exception as e:
            return self.write_response(resp=IllegalJsonBody, msg=str(e))

        # 判断 session 有效性
        redis_key = "S:{}".format(params["sessionId"])
        try:
            face_used, alive_used = await redis.redis_pool.hmget(redis_key, redis.FACE_USED, redis.ALIVE_USED)
            assert face_used is not None and face_used.isdigit(
            ), "sessionId {} not exists, may be expired, refresh it".format(params["sessionId"])
            assert int(face_used) < Config.SESSION_FACE_MAX_USED, "sessionId has been used too much, refresh it"
            assert int(alive_used) < Config.SESSION_ALIVE_MAX_USED, "sessionId has been used too much, refresh it"
        except Exception as e:
            return self.write_response(resp=IllegalSession, msg=str(e))

        # 执行人脸检测逻辑
        data = {"errorMsg": "", "errorType": 0, "transactionId": OssService.make_transaction_id()}
        try:
            image, error_msg, error_type, _ = await FaceService.face_check(params["resource"], params["actions"][0])
            await redis.redis_pool.hset(redis_key, redis.FACE_USED, int(face_used) + 1)  # 增加识别次数
            data["errorMsg"] = error_msg
            data["errorType"] = error_type

            # 埋点，保存 prepare 输入数据
            if random.random() < Config.FEATURE_SAVE_DATA_RATE:
                prepare_path = OssService.get_prepare_path(params["business"], data["transactionId"])
                save_data = {"error_msg": error_msg, "error_type": error_type, "image": params["resource"]}
                OssService.upload_mid_value(prepare_path, save_data)
        except BaseResponse as e:
            return self.write_response(resp=e)
        except Exception as e:
            logging.error(traceback.format_exc())
            return self.write_response(resp=ServerProcessingError, msg=str(e))
        return self.write_response(data=data)


class FaceAliveScoreHandler(BasicHandler):
    schema = Schema(
        {
            Required("actions"): All([int], Length(min=1, max=1)),  # 1=眨眼，2=张嘴, 3=摇头
            Required("sessionId"): All(str, Length(min=1)),
            Required("resource"): All(str, Length(min=1)),
            Optional("userId"): All(str, Length(min=1)),
            Optional("baseImage"): All(str, Length(min=1)),
            Optional("riskScore", default=""): str,
            Required("sign"): All(str, Length(min=1)),
            Required("business"): All(str, In(Config.SUPPORTED_BUSINESS))
        },
        extra=REMOVE_EXTRA)

    async def post(self):
        '''
        传入视频、动作类型数组、sessionId、原始图片/用户id
        返回活体检测结果及相似度
        sign: 视频和sessionId拼接后的 md5 签名
        '''
        # 校验参数
        try:
            q = self.get_json_args()
            params = self.schema(q)
            for i in params["actions"]:
                assert i in Config.ALIVE_ACTION_TYPES, "illegal action {}".format(i)
            assert "baseImage" in params or "userId" in params, "baseImage and userId must exist one"

        # 执行算法逻辑
        except Exception as e:
            return self.write_response(resp=IllegalJsonBody, msg=str(e))

        # 判断 session 有效性
        redis_key = "S:{}".format(params["sessionId"])
        try:
            face_used, alive_used = await redis.redis_pool.hmget(redis_key, redis.FACE_USED, redis.ALIVE_USED)
            assert face_used is not None and face_used.isdigit(
            ), "sessionId {} not exists, may be expired, refresh it".format(params["sessionId"])
            assert int(face_used) < Config.SESSION_FACE_MAX_USED, "sessionId has been used too much, refresh it"
            assert int(alive_used) < Config.SESSION_ALIVE_MAX_USED, "sessionId has been used too much, refresh it"
        except Exception as e:
            return self.write_response(resp=IllegalSession, msg=str(e))

        data = {
            "score": 0,
            "alive": 0,
            "errorMsg": "",
            "errorType": 0,
            "ismatch": 0,
            "transactionId": OssService.make_transaction_id()
        }
        try:
            '''
            # 存取中间状态
            history = await redis.redis_pool.hgetall(redis_key)  # 从 redis 获取历史数据
            alive_start = time.time()
            image, error_msg, error_type, mid_values = await FaceService.alive_check(
                params["resource"], params["actions"][0], **history)
            new_history = {k: json.dumps(v) for k, v in mid_values.__dict__.items()}
            new_history[redis.ALIVE_USED] = json.dumps(int(alive_used) + 1)
            await redis.redis_pool.hmset_dict(redis_key, new_history)  # 存入新的历史数据
            '''
            alive_start = time.time()
            image, error_msg, error_type, mid_values = await FaceService.alive_check(
                params["resource"], params["actions"][0], params["riskScore"])
            await redis.redis_pool.hset(redis_key, redis.ALIVE_USED, int(alive_used) + 1)  # 更新使用次数

            logging.info("alive detect info: {} {} {} {}".format(params.get("userId"), params["sessionId"],
                                                                 0 if image is None else 1,
                                                                 time.time() - alive_start))
            # 埋点，保存活体输入数据
            if random.random() < Config.ALIVE_SAVE_DATA_RATE:
                alive_path = OssService.get_alive_path(params["business"], data["transactionId"])
                save_data = {"error_msg": error_msg, "error_type": error_type, "video": params["resource"]}
                OssService.upload_mid_value(alive_path, save_data)
            # 活体检测失败
            if image is None:
                data["errorMsg"] = error_msg
                data["errorType"] = error_type
                return self.write_response(data=data)
            else:
                # 进行防伪判断
                score = await FaceService.get_anti_score(image=image)
                # 假视频
                if score >= Config.ANTI_FAKE_THRESHOLD:
                    data["errorMsg"] = "fake video"
                    data["errorType"] = LiveFaceDetection.ANTI_CHECK_ERROR
                    return self.write_response(data=data)
            # 活体成功
            data["alive"] = 1

            # 注册流程，仅内测使用
            user_path = OssService.get_user_path(params["business"], params.get("userId"))
            if self.request.path == "/face/alive/signup":
                assert "userId" in params, "signup must have userId"
                _, buffer = cv2.imencode('.jpg', image)
                await OssService.upload_img(user_path, io.BytesIO(buffer))
                return self.write_response(data=data)

            # 识别流程，获取底图
            base_img = params.get("baseImage")
            if base_img is None:
                # 仅内测使用
                assert await OssService.file_exists(user_path), "user {} not exists".format(params["userId"])
                binary = await OssService.download_file(user_path)
                base_img = base64.b64encode(binary).decode()
            # 人脸比对
            detected_feature = await FaceService.get_feature_value(image=image)
            base_feature = await FaceService.get_feature_value(base_img)
            data["score"] = await FaceService.get_compare_score(detected_feature, base_feature)
            data["ismatch"] = 1 if data["score"] >= Config.FEATURE_MATCH_THRESHOLD else 0
            # 保存人脸比对中间数据
            if random.random() < Config.FEATURE_SAVE_DATA_RATE:
                _, buffer = cv2.imencode('.jpg', image)
                save_data = {
                    # "detected_feature": detected_feature,
                    # "base_feature": base_feature,
                    "detected_image": base64.b64encode(buffer).decode(),
                    "base_image": base_img,
                    "score": data["score"],
                    "ismatch": data["ismatch"],
                }
                face_path = OssService.get_face_path(params["business"], data["transactionId"])
                OssService.upload_mid_value(face_path, save_data)
        except BaseResponse as e:
            return self.write_response(resp=e)
        except Exception as e:
            logging.error(traceback.format_exc())
            return self.write_response(resp=ServerProcessingError, msg=str(e))
        return self.write_response(data=data)


class FaceAliveSessionHandler(BasicHandler):
    async def get(self):
        try:
            # 动作类型数量
            number = self.get_argument("number", "1")
            number = int(number)
            assert 0 < number <= len(Config.ALIVE_ACTION_TYPES), "illegal number {}".format(number)
            business = self.get_argument("business", "")
        except Exception as e:
            return self.write_response(resp=IllegalParameter, msg=str(e))
        try:
            actions = random.sample(Config.ALIVE_ACTION_TYPES, number)
            if business in Config.ALIVE_SPECIAL_BUSINESS:
                actions = [999]
            else:
                actions = [1]  # 目前只返回 1 次眨眼
            # 生成 session id 并存储在 redis
            session_id = uuid.uuid4().hex
            key = "S:{}".format(session_id)
            f1 = redis.redis_pool.hmset(key, redis.FACE_USED, 0, redis.ALIVE_USED, 0)  # 使用次数
            f2 = redis.redis_pool.expire(key, Config.SESSION_EXPIRE_SECONDS)
            await asyncio.gather(f1, f2)
            return self.write_response(data={"actions": actions, "sessionId": session_id})
        except Exception as e:
            logging.error(traceback.format_exc())
            return self.write_response(resp=ServerProcessingError, msg=str(e))


class FaceAntiScoreHandler(BasicHandler):
    schema = Schema({Required("url"): Url(), Required("business"): All(str, In(Config.SUPPORTED_BUSINESS))})

    async def get(self):
        try:
            url = self.get_argument("url", "")
            business = self.get_argument("business", "")
            self.schema({"url": url, "business": business})
            # pro 环境服务器需要内网访问图片
            if Config.APP.get("debug", True) is False:
                url = url.replace("oss-cn-hangzhou.aliyuncs.com", "oss-cn-hangzhou-internal.aliyuncs.com")
        except Exception as e:
            return self.write_response(resp=IllegalParameter, msg=str(e))

        try:
            score = await FaceService.get_anti_score(url)
            self.write_response(data={"score": float(score), "ifFake": 1 if score >= Config.ANTI_FAKE_THRESHOLD else 0})
        except Exception as e:
            logging.error(traceback.format_exc())
            return self.write_response(resp=ServerProcessingError, msg=str(e))


class FaceAliveTrackHandler(BasicHandler):
    '''
    返回存储在 oss 的埋点信息，存储的是 json.dumps 之后的下列内容
    save_data = {
        "detected_image": base64.b64encode(buffer).decode(),
        "base_image": base_img,
        "score": data["score"],
        "ismatch": data["ismatch"],
    }
    '''
    async def get(self):
        try:
            business = self.get_argument("business")
            transactionId = self.get_argument("transactionId")
            assert business in Config.SUPPORTED_BUSINESS, "unsupported business {}".format(business)
        except Exception as e:
            return self.write_response(resp=IllegalParameter, msg=str(e))
        try:
            face_path = OssService.get_face_path(business, transactionId)
            content = await OssService.download_file(face_path)
            data = json.loads(content)
            return self.write_response(data={
                "detectedImage": data["detected_image"],
                "score": data["score"],
                "ismatch": data["ismatch"],
            })
        except Exception as e:
            logging.error(traceback.format_exc())
            return self.write_response(resp=ServerProcessingError, msg=str(e))
