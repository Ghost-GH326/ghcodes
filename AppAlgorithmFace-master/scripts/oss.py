import os
import base64
import oss2
import json


# {business}/user/，用户注册的图像
# {business}/prepare/，准备阶段的图像
# {business}/face/，人脸比对的中间结果
# {business}/alive/，活体检测的入参
def print_keys():
    r = bucket.list_objects()
    for item in r.object_list:
        print(item.key)
        # bucket.delete_object(item.key)


def get_urls(url_type, dt_str):
    types = ("alive", "face", "prepare")
    assert url_type in types, "url_type must in {}".format(types)
    marker = None
    while True:
        r = bucket.list_objects(prefix="dev/TEST/{}/{}".format(url_type, dt_str), marker=marker, max_keys=1000)
        for item in r.object_list:
            url = bucket.sign_url("GET", item.key, 3600)
            print(item.key, url)
        marker = r.next_marker
        if not r.is_truncated:
            break


# 拉取 prepare 图像
def load_prepare_file(file_path, image_path):
    with open(file_path) as f:
        content = f.read()
    data = json.loads(content)
    print(data["error_msg"], data["error_type"])
    binary = base64.b64decode(data["image"].encode())
    with open(image_path, "wb") as f:
        f.write(binary)


# 拉取活体检测视频
def load_alive_file(file_path, video_path):
    with open(file_path) as f:
        content = f.read()
    data = json.loads(content)
    print(data["error_msg"], data["error_type"])
    binary = base64.b64decode(data["video"].encode())
    with open(video_path, "wb") as f:
        f.write(binary)


# 拉取人脸认证底图和视频中抽取的图片
def load_face_file(file_path, base_image_path, detected_image_path):
    with open(file_path) as f:
        content = f.read()
    data = json.loads(content)
    print(data["score"], data.get("ismatch"))
    with open(base_image_path, "wb") as f:
        binary = base64.b64decode(data["base_image"].encode())
        f.write(binary)
    with open(detected_image_path, "wb") as f:
        binary = base64.b64decode(data["detected_image"].encode())
        f.write(binary)


def upload_img(key, local_path):
    bucket.put_object_from_file(key, local_path, headers={"Content-Type": "image/jpeg"})
    url = bucket.sign_url("GET", key, 3600 * 24 * 365)
    print(url)


def download_models(prefix):
    # prefix = "AppAlgorithmFace/models/abc/"
    folders = bucket.list_objects(prefix=prefix)
    for f in folders.object_list:
        if f.key.endswith("/"):
            continue
        file_path = os.path.join(*f.key.split("/")[1:])
        folder_path = file_path.rsplit("/", 1)[0]
        print(f.key, folder_path, file_path)
        os.makedirs(folder_path, exist_ok=True)
        bucket.get_object_to_file(f.key, file_path)


if __name__ == "__main__":
    # get_urls("face", "20200623194")
    # load_alive_file("/Users/wxy/Desktop/dev_TEST_alive_20200623194606a33a59ae7bd240b9bf8582929530369c", "/Users/wxy/Desktop/1.webm")
    # load_prepare_file("/Users/wxy/Desktop/dev_TEST_prepare_20200623194603af0ba3291b754f6bb156ce5e7983d570", "/Users/wxy/Desktop/1.jpg")
    # load_face_file("/Users/wxy/Desktop/dev_TEST_face_20200623194606a33a59ae7bd240b9bf8582929530369c", "/Users/wxy/Desktop/base.jpg", "/Users/wxy/Desktop/detect.jpg")

