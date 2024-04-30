import os
os.chdir('/home/guohao826/AppAlgorithmFace')
import cv2
import uuid
import time
import logging
import ffmpeg
import subprocess
from src.config import Config
from src.service.face_alg.detect_inference import DetectInference

from src.service.face_alg.service_demo import PthExtractor

from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm

## 加载模型
detector = DetectInference(Config.DETECT_MODEL_PATH, Config.DETECT_GPUID, Config.DETECT_INPUT_SHAPE,
                        Config.DETECT_THRESHOLD)
feature_model = PthExtractor(Config.FEATURE_MODEL_PATH,
                        Config.FEATURE_MODEL_CONFIG,
                        Config.FEATURE_GPUID,
                        batch_size=Config.FEATURE_BATCH_SIZE,
                        im_shape=Config.FEATURE_INPUT_SHAPE)

def face_detector(video,interval,face_time,fps,video_path,filtvideos_folder):
    ## 处理视频，找到包含目标人脸的时间片段
    # 记录有人脸的帧数
    frame_count = 0
    frames_with_face = 0
    # 遍历视频的每一帧
    while True:
        ret, frame = video.read()
        # 如果没有帧了，退出循环
        if not ret:
            break
        face_detected = False
        # 抽帧
        if frame_count % interval == 0:
            ## 人脸检测
            bboxes,lads = detector.detect([frame])
            bounding_boxes = bboxes[0]
            if len(bounding_boxes) > 0 and len(bounding_boxes) <= 3:     #需要设置出现人脸数，修改此处/ 增加 and len(bounding_boxes) <= 3（yifeng）
                face_detected = True
        # 对有人脸的帧进行处理
        if face_detected:
            # # 保存人脸帧
            # output_folder = "/home/guohao826/AppAlgorithmFace/material/output_frame"
            # frame_filename = f"fightclub_{frames_with_face}.jpg"
            # frame_path = os.path.join(output_folder, frame_filename)
            # cv2.imwrite(frame_path, frame)
            
            frames_with_face += 1
        frame_count += 1
    # 释放视频流
    video.release()
    
    if frames_with_face * interval / fps > face_time: # 将fps/interval 替换成 interval / fps （yifeng）
        command = f'cp {video_path} {filtvideos_folder}'
        subprocess.run(command, shell=True)

if __name__ == "__main__":
    ## 测试代码，使用时替换成实际的文件路径和参数
    # videos_folder = '/home/guohao826/Video_segmentation_by_time_10s/Movie10_segment'
    videos_folder = '/home/guohao826/short_videos_400'
    # filtvideos_folder = '/home/guohao826/AppAlgorithmFace/material/filt_videos_movie10'
    filtvideos_folder = '/home/guohao826/AppAlgorithmFace/material/filt_videos_shortvideo2'
    frame_interval = 0.5  # 设置帧抽取的时间间隔
    face_time = 8  # 设置人脸出现时间秒数
        # 打开视频文件
    videos = os.listdir(videos_folder)
    for video_name in tqdm(videos):
        video_path = os.path.join( videos_folder,video_name)
        video = cv2.VideoCapture(video_path)
        ff_video = ffmpeg.probe(video_path)
        # 获取视频的帧率
        # fps = int(video.get(cv2.CAP_PROP_FPS))
        fps = int(int(ff_video['streams'][0]['r_frame_rate'].split('/')[0]) / int(ff_video['streams'][0]['r_frame_rate'].split('/')[1]))
        # 获取视频的宽度和高度
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 设置抽帧
        # print(f"帧率：{fps}")
        interval = int(fps * frame_interval)
        face_detector(video,interval,face_time,fps,video_path,filtvideos_folder)




# '''
# Descripttion: 
# version: 1.0
# Author: lbinb
# Date: 2020-09-28 16:57:16
# LastEditors: lbinb
# LastEditTime: 2023-07-24 14:27:17
# '''
# # -*- encoding=utf-8 -*-
# from functools import reduce
# from PIL import Image,ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# from tqdm import tqdm
# import cv2

# from functools import reduce
# from PIL import Image,ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# from io import BytesIO
# import base64
# import json

# def phash(img):
#     """
#     :param img: 图片
#     :return: 返回图片的局部hash值
#     """
#     img = img.resize((16, 16), Image.ANTIALIAS).convert('L')
#     avg = reduce(lambda x, y: x + y, img.getdata()) / 256.
#     hash_value=reduce(lambda x, y: x | (y[1] << y[0]), enumerate(map(lambda i: 0 if i < avg else 1, img.getdata())), 0)
#     return hash_value

# #计算两个图片相似度函数局部敏感哈希算法
# def phash_img_similarity(img1_h,img2_h):
#     """
#     :param img1: 图片1的hash值
#     :param img2: 图片2的hash值
#     :return: 图片相似度
#     """
#     # 计算汉明距离
#     distance = bin(img1_h ^ img2_h).count('1')
#     similary = 1 - distance / max(len(bin(img1_h)), len(bin(img2_h)))
#     return similary
    
# def compare_pairs(img1,img2):
#     img1_hash = phash(img1)
#     img2_hash = phash(img2)
#     #print(type(img1_hash))
#     img_similarity = phash_img_similarity(img1_hash,img2_hash)
#     return img_similarity

# def getImageVar(img_path):
#     image = cv2.imread(img_path)
#     img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     imageVar = cv2.Laplacian(img2gray, cv2.CV_64F).var()
#     return imageVar

# def image_to_base64(image_path):
#     img = Image.open(image_path)
#     output_buffer = BytesIO()
#     img.save(output_buffer, format='JPEG')
#     byte_data = output_buffer.getvalue()
#     base64_str = base64.b64encode(byte_data)
#     return base64_str


# if __name__ == '__main__':

#     img_path = '/Users/lbinb/hellobike/f0_180.jpg'
#     img2_path = '/Users/lbinb/hellobike/f0_235.jpg'
#     #img2_path = '/Users/lbinb/hellobike/caad70dd37fc44188fe47ec9023a4a57.jpeg'
    
#     img1 = Image.open(img_path)
#     img1_hash = phash(img1)
#     # print(img1_hash)
#     #print(img1_hash)
#     #img_phash = phash(img1) 
#     img2 = Image.open(img2_path)
#     img2_phash = phash(img2)
#     # img2_phash = 1234172358074279319006949602245063871915004682697970415273246719
#     img_similarity = phash_img_similarity(img1_hash,img2_phash)
#     print(img1_hash)
#     print(img2_phash)
#     #img_similarity = compare_pairs(img1,img2) 
#     print(img_similarity)
#     # a = image_to_base64(img_path)
#     # print(a)