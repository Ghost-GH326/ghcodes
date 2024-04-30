# -*- coding: utf-8 -*-
'''
调用 http 接口进行人脸算法测试
1. 安装依赖：pip install requests
2. 修改 HOST 值
3. 执行脚本，python client.py
'''
from __future__ import print_function
import sys
import base64
import requests
import hashlib

HOST = "http://127.0.0.1:9185"
# HOST = "http://10.111.70.7:9185"  # dev
# HOST = "http://10.111.12.135:9185"  # fat


def upload_video(video_path, content_type):
    url = "{}/test/video/upload".format(HOST)
    multiple_files = [
        ('video', ('test.wemb', open(video_path, 'rb'), content_type)),
    ]
    r = requests.post(url, files=multiple_files)
    return r.json()


def get_feature(image_path):
    url = "{}/test/feature".format(HOST)
    with open(image_path, "rb") as f:
        binary = f.read()
    data = {"image": base64.b64encode(binary).decode()}
    r = requests.post(url, json=data)
    return r.json()


def signup(video_path, session_id, action_type):
    url = "{}/face/alive/signup".format(HOST)
    with open(video_path, "rb") as f:
        video = base64.b64encode(f.read()).decode()
    data = {
        "resource": video,
        "sessionId": session_id,
        "actions": [int(action_type)],
        "userId": "09970",
        "business": "TEST",
        "sign": hashlib.md5((video + session_id).encode()).hexdigest()
    }
    r = requests.post(url, json=data)
    return r.json()


def recognize(video_path, image_path, session_id, action_type):
    url = "{}/face/alive/recognize".format(HOST)
    with open(video_path, "rb") as f:
        video = base64.b64encode(f.read()).decode()
    with open(image_path, "rb") as f:
        image = base64.b64encode(f.read()).decode()
    data = {
        "resource": video,
        "sessionId": session_id,
        "actions": [int(action_type)],
        "baseImage": image,
        "business": "TEST",
        "riskScore": "",
        "sign": hashlib.md5((video + session_id).encode()).hexdigest()
    }
    r = requests.post(url, json=data)
    return r.json()


def prepare(image_path, session_id, action_type):
    url = "{}/face/alive/prepare".format(HOST)
    with open(image_path, "rb") as f:
        image = base64.b64encode(f.read()).decode()
    data = {
        "resource": image,
        "sessionId": session_id,
        "actions": [int(action_type)],
        "business": "TEST",
        "sign": hashlib.md5((image + session_id).encode()).hexdigest()
    }
    r = requests.post(url, json=data)
    return r.json()


class Wrapper:
    def __init__(self, func, arg_len, useage):
        self.func = func
        self.arg_len = arg_len
        self.useage = useage


if __name__ == "__main__":
    FUNC_MAP = {
        "get_feature":
        Wrapper(get_feature, 1, "python client.py get_feature /path/to/image"),
        "upload_video":
        Wrapper(upload_video, 2, "python client.py upload_video /path/to/video content_type"),
        "signup":
        Wrapper(signup, 3,
                "python client.py signup /path/to/video session_id action_type(1 for eye, 2 for mouse, 3 for shake)"),
        "prepare":
        Wrapper(prepare, 3,
                "python client.py prepare /path/to/image session_id action_type(1 for eye, 2 for mouse, 3 for shake)"),
        "recognize":
        Wrapper(
            recognize, 4,
            "python client.py recognize /path/to/video /path/to/image session_id action_type(1 for eye, 2 for mouse, 3 for shake)"
        ),
    }
    # 检查脚本参数
    if len(sys.argv) < 2 or FUNC_MAP.get(sys.argv[1]) is None:
        print("error! useage: python client.py {}".format(list(FUNC_MAP.keys())))
    else:
        wrapper = FUNC_MAP[sys.argv[1]]
        args = sys.argv[2:]
        # 检查函数参数
        if len(args) != wrapper.arg_len:
            print("error! useage: {}".format(wrapper.useage))
            sys.exit(1)
        # 执行
        print(wrapper.func(*args))
