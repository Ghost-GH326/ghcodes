import json
import uuid
import logging
import asyncio
import oss2
from datetime import datetime
import concurrent.futures as cf
from src.config import Config


class OssService:
    @classmethod
    def init(cls):
        auth = oss2.Auth(Config.OSS_ACCESS_KEY_ID, Config.OSS_ACCESS_KEY_SECRET)
        cls.bucket = oss2.Bucket(auth, Config.OSS_ENDPOINT, Config.OSS_BUCKET_NAME)
        cls.executors = cf.ThreadPoolExecutor(max_workers=Config.OSS_EXECUTORS_PER_PROCESS)
        cls.loop = asyncio.get_event_loop()

    @classmethod
    def sign_path(cls, path):
        if cls.bucket.object_exists(path):
            return cls.bucket.sign_url("GET", path, Config.OSS_URL_EXPIRES)
        else:
            logging.info("oss {} not exists".format(path))
            return None

    @classmethod
    def upload_mid_value(cls, key, msg_dict):
        try:
            future = cls.loop.run_in_executor(cls.executors, cls.bucket.put_object, key, json.dumps(msg_dict))
            future.add_done_callback(cls._check_result)
        except Exception as e:
            logging.error(e)

    @classmethod
    async def upload_img(cls, key, binary):
        r = await cls.loop.run_in_executor(cls.executors, cls.bucket.put_object, key, binary,
                                           {"Content-Type": "image/jpeg"})
        assert r.status == 200, "upload image error {}".format(r.status)
        return cls.bucket.sign_url("GET", key, Config.OSS_URL_EXPIRES)

    @classmethod
    def _check_result(cls, future):
        try:
            r = future.result()
            assert r.status == 200, "save data, oss error {}".format(r.status)
            logging.debug("check result ok")
        except Exception as e:
            logging.error(e)

    @classmethod
    async def download_file(cls, key):
        r = await cls.loop.run_in_executor(cls.executors, cls.bucket.get_object, key)
        assert r.status == 200, "download file {} error {}".format(key, r.status)
        return r.read()

    @classmethod
    async def file_exists(cls, key):
        return await cls.loop.run_in_executor(cls.executors, cls.bucket.object_exists, key)

    @classmethod
    def get_user_path(cls, business, name):
        return "{}/{}/user/{}".format(Config.OSS_FOLDER_NAME, business, name)

    @classmethod
    def get_alive_path(cls, business, name):
        return "{}/{}/alive/{}".format(Config.OSS_FOLDER_NAME, business, name)

    @classmethod
    def get_face_path(cls, business, name):
        return "{}/{}/face/{}".format(Config.OSS_FOLDER_NAME, business, name)

    @classmethod
    def get_prepare_path(cls, business, name):
        return "{}/{}/prepare/{}".format(Config.OSS_FOLDER_NAME, business, name)

    @classmethod
    def make_transaction_id(cls):
        date_str = datetime.now().strftime("%Y%m%d%H%M%S")
        uid = uuid.uuid4().hex
        return "{}{}".format(date_str, uid)
