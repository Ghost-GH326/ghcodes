import os
os.chdir('/home/guohao826/AppAlgorithmFace')
import cv2, ffmpeg, argparse, torch, subprocess, requests
import numpy as np
from tqdm import tqdm
from src.config import Config
from src.service.face_alg.detect_inference import DetectInference
from src.service.face_alg.service_demo import PthExtractor, detect_align, get_sim, detect_align_video
from src.util.decorator import catch
from functools import reduce
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from datetime import timedelta

## 加载模型
detector = DetectInference(Config.DETECT_MODEL_PATH, Config.DETECT_GPUID, Config.DETECT_INPUT_SHAPE,
                        Config.DETECT_THRESHOLD)
feature_model = PthExtractor(Config.FEATURE_MODEL_PATH,
                        Config.FEATURE_MODEL_CONFIG,
                        Config.FEATURE_GPUID,
                        batch_size=Config.FEATURE_BATCH_SIZE,
                        im_shape=Config.FEATURE_INPUT_SHAPE)

def options():
    parser = argparse.ArgumentParser(description=' ')

    parser.add_argument('')
    
    pass


def face_detector(frame):
    face_detected = False
## 人脸检测
    bboxes,lads = detector.detect([frame])
    bounding_boxes = bboxes[0]
    if len(bounding_boxes) > 0 :     #需要设置出现人脸数，修改此处
        face_detected = True
        # 对有人脸的帧进行处理
    # if face_detected:
    #     pass
    #     # # 保存人脸帧
    #     # output_folder = "/home/guohao826/AppAlgorithmFace/material/output_frame"
    #     # frame_filename = f"fightclub_{frames_with_face}.jpg"
    #     # frame_path = os.path.join(output_folder, frame_filename)
    #     # cv2.imwrite(frame_path, frame)
    return face_detected

def generate_video_time(total_seconds):
    time_format = str(timedelta(seconds=total_seconds))
    hms = f"{int(time_format.split(':')[0]):02d}:{int(time_format.split(':')[1]):02d}:{int(time_format.split(':')[2]):02d}"
    return hms

def down_url2bytes(url):
    resp = requests.get(url)
    binary = resp.content
    video_bytes = bytes(binary)
    return video_bytes

def write_bytes2video(bytes,save_path):
    wfile = open(save_path, 'wb')
    wfile.write(bytes)
    wfile.close()

def down_video(url,save_path):
    video_bytes = down_url2bytes(url)
    write_bytes2video(video_bytes,save_path)