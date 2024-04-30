import streamlit as st
import cv2, io, os, time
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
net = YOLO("/home/guohao826/yolov8m-pose.pt")
# Define the pairs of joints to connect for the skeleton
POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4],  # Trunk
              [1, 5], [5, 6], [6, 7], [1, 11], [11, 12], [12, 13],  # Arms
              [1, 8], [8, 9], [9, 10]]  # Legs
## Function to draw skeleton on the frame
def draw_skeleton(frame, results):
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xywh.cpu().numpy().astype(int)[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2 + x1), int(y2 + y1)
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                ## Draw skeleton keypoints
                if result.keypoints is not None:
                    kp_data = result.keypoints.data.cpu().numpy()[0]
                    for kp in kp_data:
                        x, y, conf = int(kp[0]), int(kp[1]), kp[2]
                        if conf > 0.5:
                            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
                        
                    ## Draw skeleton connections
                    skeleton_connections = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
                                            [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 16], [9, 3], [4, 3], [4, 2]]
                    for connection in skeleton_connections:
                        start_kp, end_kp = connection
                        start_x, start_y = int(kp_data[start_kp-1][0]), int(kp_data[start_kp-1][1])
                        end_x, end_y = int(kp_data[end_kp][0]), int(kp_data[end_kp][1])
                        if start_x != 0 and start_y != 0 and end_x != 0 and end_y != 0:
                            cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
                    
    return frame

def save_video_to_file(video_bytes, filename):
    with open(filename, "wb") as f:
        f.write(video_bytes.getbuffer())
    st.success(f"Video saved as {filename}")

# Function to process the video
def process_video(video_file):
    
    # Your video processing code here
    # ...
    # Return the processed output
    return 92

    # Function to preprocess the video
def preprocess_video(video_path):
    ## Open the video file
    cap = cv2.VideoCapture(video_path)

    ## Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ## Create a video writer to save the output
    output_path = '/home/guohao826/CoRe/Mydata/outputVideos/output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    ## Process each frame of the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        ## Detect skeletons in the frame
        results = net.predict(frame)
        
        ## Draw skeleton on the frame
        output_frame = draw_skeleton(frame, results)
        
        ## Write the output frame to the video file
        out.write(output_frame)
        
    ## Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    outputVideo = open(output_path, "rb")
    outputVideoBytes = outputVideo.read()
    return outputVideoBytes

# Streamlit app
def app():
    st.title("Video Action Quality Assessment App")

    # Upload video
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        video_bytes = io.BytesIO(uploaded_video.read())
        st.video(video_bytes)

        # Save video to file
        filename = os.path.join("/home/guohao826/CoRe/Mydata/uploads", uploaded_video.name)
        save_video_to_file(video_bytes, filename)
        # Preview video
        # video_bytes = uploaded_video.read()
        # st.video(video_bytes)

        # Preprocess video
        if st.button("Action recognition"):
            with st.spinner("processing video..."):
                preprocessed_video_bytes = preprocess_video(filename)

            # Display preprocessed video
            st.header("processed Video")
            # outputVideo = open('/home/guohao826/CoRe/Mydata/outputVideos/output.mp4', "rb")
            # preprocessed_video_bytes = outputVideo.read()
            st.video(preprocessed_video_bytes)

        # Process video
        if st.button("Assess Video"):
            with st.spinner("Processing video..."):
                processed_output = process_video(video_bytes)

            # Display processed output
            st.header("Predict Score")
            st.write(processed_output)

if __name__ == "__main__":
    app()
