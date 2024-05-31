import streamlit as st
import os, io
import time

# Set the title of the page
st.title("视频动作识别和动作质量评估系统")

# Create a file uploader for the user to upload a video
uploaded_video = st.file_uploader("上传视频", type=["mp4", "avi", "mov"])

# Create a folder to store the uploaded video
video_folder = "uploaded_videos"
if not os.path.exists(video_folder):
    os.makedirs(video_folder)

# If the user has uploaded a video, save it to the folder
if uploaded_video is not None:
    video_bytes = io.BytesIO(uploaded_video.read())
    st.video(video_bytes)
    video_path = os.path.join(video_folder, uploaded_video.name)
    with open(video_path, "wb") as f:
        f.write(video_bytes.getbuffer())
    st.success(f"视频已上传：{uploaded_video.name}")

    # Create a button to start the action recognition process
    if st.button("动作识别"):
        # Simulate the action recognition process (replace with your actual code)
        st.write("正在处理视频...")
        time.sleep(1)  # Simulate a 10-second processing time
        st.write("视频处理完成！")

        # Generate 27 different videos (replace with your actual code)
        video_titles = [f"第{i}节" for i in range(1, 28)]
        videos = []
        for title in video_titles:
            # Simulate generating a video (replace with your actual code)
            video_path = os.path.join(video_folder, f"{title}.mp4")
            with open(video_path, "wb") as f:
                f.write(b"Simulated video content")
            videos.append(video_path)

        # Create a navigation menu for the videos
        selected_video = st.experimental_memo.get("selected_video", None)
        if selected_video is None:
            st.experimental_memo.set("selected_video", 0)
        st.write(f"当前选择：{video_titles[selected_video]}")

        # Display the selected video
        if selected_video is not None:
            st.write(video_titles[selected_video])
            st.video(videos[selected_video])

            # Create a button to evaluate the action quality
            if st.button(f"动作质量评价 ({video_titles[selected_video]})"):
                # Simulate the action quality evaluation process (replace with your actual code)
                st.write("正在评估动作质量...")
                time.sleep(1)  # Simulate a 5-second processing time
                st.write("动作质量评估结果：Excellent！")
        else:
            st.stop()
