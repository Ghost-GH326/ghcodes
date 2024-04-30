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

from src.service.face_alg.service_demo import PthExtractor, detect_align, get_sim, detect_align_video
from src.util.decorator import catch

class FaceRecognition:
    def __init__(self, video_path, images_folder, frame_interval=1, video_threshold=5):
        self.video_path = video_path
        self.images_folder = images_folder
        self.frame_interval = frame_interval
        self.video_threshold = video_threshold

        ## 加载模型
        self.detector = DetectInference(Config.DETECT_MODEL_PATH, Config.DETECT_GPUID, Config.DETECT_INPUT_SHAPE,
                               Config.DETECT_THRESHOLD)
        self.feature_model = PthExtractor(Config.FEATURE_MODEL_PATH,
                             Config.FEATURE_MODEL_CONFIG,
                             Config.FEATURE_GPUID,
                             batch_size=Config.FEATURE_BATCH_SIZE,
                             im_shape=Config.FEATURE_INPUT_SHAPE)

    def process_video(self):
        ## 处理视频，找到包含目标人脸的时间片段
        # 打开视频文件
        video = cv2.VideoCapture(self.video_path)
        images = os.listdir(self.images_folder)
        ff_video = ffmpeg.probe(self.video_path)
        # 获取视频的帧率
        # fps = int(video.get(cv2.CAP_PROP_FPS))
        fps = int(int(ff_video['streams'][0]['r_frame_rate'].split('/')[0]) / int(ff_video['streams'][0]['r_frame_rate'].split('/')[1]))
        # 获取视频的宽度和高度
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 记录有人脸的帧数
        frame_count = 0
        frames_with_face = 0
        frames_with_images = {}
        # 设置抽帧
        print(f"帧率：{fps}")
        interval = int(fps) * frame_interval
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
                bboxes,lads = self.detector.detect([frame])
                bounding_boxes = bboxes[0]
                if len(bounding_boxes) >0 and len(bounding_boxes)<3:
                    face_detected = True
            # 对有人脸的帧进行处理
            if face_detected:
                # # 保存人脸帧
                # output_folder = "/home/guohao826/AppAlgorithmFace/material/output_frame"
                # frame_filename = f"fightclub_{frames_with_face}.jpg"
                # frame_path = os.path.join(output_folder, frame_filename)
                # cv2.imwrite(frame_path, frame)

                filename_frame = uuid.uuid4().hex
                logging.debug("detect start: %s", time.time())
                key_imarray_map = detect_align_video(self.detector, {filename_frame: frame}, align_shape=Config.DETECT_ALIGN_SHAPE)
                logging.debug("detect done: %s", time.time())
                feature_map = self.feature_model.inference(key_imarray_map)
                logging.debug("feature done: %s", time.time())
                detect_feature = feature_map
                
                for imagename in images:
                    try:
                        image_path = os.path.join(images_folder, imagename)
                        image = cv2.imread(image_path)
                        ## 根据特征计算相似度得分
                        filename_image = uuid.uuid4().hex
                        logging.debug("detect start: %s", time.time())
                        key_imarray_map = detect_align(self.detector, {filename_image: image}, align_shape=Config.DETECT_ALIGN_SHAPE)
                        logging.debug("detect done: %s", time.time())
                        feature_map = self.feature_model.inference(key_imarray_map)
                        logging.debug("feature done: %s", time.time())
                        base_feature = feature_map[filename_image]
                        ## 人脸比对
                        for key in detect_feature:
                            data_score = get_sim(detect_feature[key], base_feature)
                            data_ismatch = True if data_score >= Config.FEATURE_MATCH_THRESHOLD else False
                            # 保存对应人脸的时间片段
                            if data_ismatch:
                                if imagename not in frames_with_images:
                                    frames_with_images.setdefault(imagename, [frame_count, ])
                                else:
                                    frames_with_images[imagename].append(frame_count)

                                # #保存帧
                                # output_folder = "/home/guohao826/AppAlgorithmFace/material/output_frame"
                                # frame_filename = f"{imagename}_{frame_count/fps}.jpg"
                                # frame_path = os.path.join(output_folder, frame_filename)
                                # cv2.imwrite(frame_path, frame)
                    except:
                        print(imagename)
                frames_with_face += 1
            frame_count += 1
        # 释放视频流
        video.release()
        print(frame_count)

        ## 可视化处理
        for key in frames_with_images:
            timelines = []
            start = frames_with_images[key][0]/fps
            for i in range(0,len(frames_with_images[key])):
                if i == len(frames_with_images[key])-1 or ((frames_with_images[key][i] + (self.frame_interval * fps) != frames_with_images[key][i+1]) and ((frames_with_images[key][i+1] - frames_with_images[key][i]) > (10 * fps))):
                    end = frames_with_images[key][i]/fps
                    if start >= 60:
                        minute_s = f'{int(start // 60)}' if start // 60 >= 10 else f'0{int(start // 60)}'
                        second_s = f'{int(start % 60)}' if start % 60 >= 10 else f'0{int(start % 60)}'
                    else:
                        minute_s = '00'
                        second_s = f'{int(start)}' if start >= 10 else f'0{int(start)}'
                    if end >= 60:
                        minute_e = f'{int(end // 60)}' if end // 60 >=10 else f'0{int(end // 60)}'
                        second_e = f'{int(end % 60)}' if end % 60 >=10 else f'0{int(end % 60)}'
                    else:
                        minute_e = '00'
                        second_e = f'{int(end)}' if end >=10 else f'0{int(end)}'
                    timelines.append((f'{minute_s}:{second_s}', f'{minute_e}:{second_e}'))
                    ## 剪辑视频片段
                    output_path = f'/home/guohao826/AppAlgorithmFace/material/output_video/zhenhuan04/04_{key}_{minute_s}_{second_s}to{minute_e}_{second_e}.mp4'
                    if end - start > video_threshold:
                        command = f'ffmpeg -ss 00:{minute_s}:{second_s} -i {self.video_path} -t {int(end - start)} -c copy {output_path}'
                        subprocess.run(command, shell=True)
                    if i != len(frames_with_images[key])-1:
                        start = frames_with_images[key][i+1]/fps
            print(f"包含{key}的片段有(分:秒):{timelines}")

if __name__ == "__main__":
    ## 测试代码，使用时替换成实际的文件路径和参数
    video_path = '/home/guohao826/AppAlgorithmFace/material/input_video/zhenhuan1/zhenhuan3-5/04.mp4'
    images_folder = "/home/guohao826/AppAlgorithmFace/material/images/zhenhuan1" # 用于比对的目标人脸文件夹
    frame_interval = 1  # 设置帧抽取的时间间隔
    video_threshold = 5 # 设置最小时间片段

    face_recognition = FaceRecognition(video_path, images_folder, frame_interval, video_threshold)

    # 分别处理视频和图片
    face_recognition.process_video()