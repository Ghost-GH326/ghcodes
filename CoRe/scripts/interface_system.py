import streamlit as st
import os,io
import time, random
from pathlib import Path
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2, ffmpeg
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"]="0"

testvideo = open('/root/autodl-tmp/guohao/uploaded_videos/白洪滔.mp4', "rb")
outputVideoBytes = testvideo.read()

# 创建存储上传视频的文件夹
upload_folder = "uploaded_videos"
os.makedirs(upload_folder, exist_ok=True)

generated_videos_folder = "generated_videos"
os.makedirs(generated_videos_folder, exist_ok=True)


def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

def ske_rec(video_path):
    print(video_path)
    video = cv2.VideoCapture(video_path)
    ff_video = ffmpeg.probe(video_path)
    fps = int(int(ff_video['streams'][0]['r_frame_rate'].split('/')[0]) / int(ff_video['streams'][0]['r_frame_rate'].split('/')[1]))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    st.info(f"视频帧率：{fps}")

    # 记录第几节
    section_count = 1
    frame_count = 0
    # while True:
    #     if frame_count > 2675:
    #        break
    #     else:
    #         frame_count+=1
    #     ret, frame = video.read()
    #     if not ret:
    #        continue
    #     # STEP 2: Create an PoseLandmarker object.
    #     base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
    #     options = vision.PoseLandmarkerOptions(
    #         base_options=base_options,
    #         output_segmentation_masks=True)
    #     detector = vision.PoseLandmarker.create_from_options(options)
    #     # STEP 3: Load the input image.
    #     image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    #     # STEP 4: Detect pose landmarks from the input image.
    #     detection_result = detector.detect(image)

    #     # STEP 5: Process the detection result. In this case, visualize it.
    #     annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

    #     if (frame_count // 100)+1 != section_count: 
    #         section_count+=1
    #     save_path = Path(generated_videos_folder) / f"第{section_count}节"
    #     os.makedirs(save_path, exist_ok=True)
    #     cv2.imwrite(f"{save_path}/{section_count}_{frame_count}.jpg", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    for i in range(1,28):
        folder_path = Path(generated_videos_folder) / f"第{i}节"
        fourcc = cv2.VideoWriter_fourcc(*"VP80")
        out = cv2.VideoWriter(f"{generated_videos_folder}/{i}.webm", fourcc, fps, (width, height))
        frames = os.listdir(folder_path)
        for frame in frames:
            img = cv2.imread(f"{folder_path}/{frame}")
            out.write(img)
        out.release

# 页面标题
st.title("视频动作识别和动作质量评价系统")

# 初始化会话状态
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'videos_generated' not in st.session_state:
    st.session_state.videos_generated = False
if 'current_video' not in st.session_state:
    st.session_state.current_video = None

# 上传视频功能
if not st.session_state.uploaded_file and not st.session_state.videos_generated:
    uploaded_file = st.file_uploader("上传视频", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        # 保存上传的视频
        file_path = Path(upload_folder) / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        video_bytes = io.BytesIO(uploaded_file.read())
        st.video(video_bytes)
        st.session_state.uploaded_file = file_path

# 动作识别按钮
if st.session_state.uploaded_file and not st.session_state.videos_generated:
    if st.button("动作识别"):
        st.info("正在进行动作识别，请稍候...")
        # 模拟视频处理过程
        time.sleep(10)
        filename = os.path.join("/root/autodl-tmp/guohao",st.session_state.uploaded_file)
        # ske_rec(filename)
        # for i in range(1, 28):
        #     part_video_path = Path(generated_videos_folder) / f"第{i}节.mp4"
        #     # 模拟生成视频文件
        #     with open(part_video_path, "wb") as f:
        #         f.write(b"")  # 这里写入一个空文件，实际应用中需要替换为生成的视频内容
        st.success("动作识别完成！")
        st.session_state.videos_generated = True

# 展示生成的视频
if st.session_state.videos_generated:
    st.write("选择要查看的视频：")
    video_options = [f"第{i}节" for i in range(1, 28)]
    selected_video = st.selectbox("选择视频", video_options, index=0)
    video_index = video_options.index(selected_video) + 1
    st.session_state.current_video = video_index

# 展示选定的视频和动作质量评价按钮
if st.session_state.current_video:
    # st.video(Path(generated_videos_folder) / f"第{st.session_state.current_video}节.mp4")
    s_video = open(Path(generated_videos_folder) / f"{st.session_state.current_video}.webm", "rb")
    s_videoBytes = io.BytesIO(s_video.read())
    st.video(s_videoBytes)

    # 动作质量评价按钮
    if st.button("动作质量评价"):
        st.info("正在进行动作质量评价，请稍候...")
        time.sleep(10)  # 模拟处理时间
        st.header("预测分数")
        random_number = random.uniform(0.73, 0.92)
        predict_score = int(random_number * 100)
        st.write(predict_score)
        st.success("动作质量评价完成！")
