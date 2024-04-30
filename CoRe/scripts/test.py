import streamlit as st
import cv2, io, os, time
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
net = YOLO("/home/guohao826/yolov8s-pose.pt")
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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                ## Draw skeleton keypoints
                if result.keypoints is not None:
                    kp_data = result.keypoints.data.cpu().numpy()[0]
                    for kp in kp_data:
                        x, y, conf = int(kp[0]), int(kp[1]), kp[2]
                        if conf > 0.8:
                            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
                        
                    ## Draw skeleton connections
                    skeleton_connections = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
                                            [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 16], [9, 3], [4, 3], [4, 2]]
                    for connection in skeleton_connections:
                        start_kp, end_kp = connection
                        start_x, start_y = int(kp_data[start_kp-1][0]), int(kp_data[start_kp-1][1])
                        end_x, end_y = int(kp_data[end_kp][0]), int(kp_data[end_kp][1])
                        cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
                    
    return frame



if __name__ == "__main__":
    cap = cv2.VideoCapture('/home/guohao826/CoRe/Mydata/uploads/001.avi')

    ## Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ## Create a video writer to save the output
    output_path = '/home/guohao826/CoRe/Mydata/outputVideos/output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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