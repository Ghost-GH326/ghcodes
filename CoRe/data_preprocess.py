import ffmpeg
import subprocess
import cv2
import os

videos_folder = '/home/guohao826/CoRe/AQA-7/Actions/sync_diving_10m'
images_folder = '/home/guohao826/CoRe/Seven/sync_diving_10m-out'

videos = os.listdir(videos_folder)
for video_name in videos:
    video_count = video_name.split('.')[0]
    frames_path = os.path.join(images_folder, video_count)
    os.mkdir(frames_path)
    frame_count = 1
    video_path = os.path.join(videos_folder, video_name)
    video = cv2.VideoCapture(video_path)
    ff_video = ffmpeg.probe(video_path)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_name = f'img_{int(frame_count):0>5d}.jpg'
        frame_path = os.path.join(frames_path, frame_name)
        cv2.imwrite(frame_path, frame)
        frame_count+=1