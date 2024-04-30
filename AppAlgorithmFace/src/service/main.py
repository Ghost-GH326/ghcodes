import os
import cv2
import sys
import logging
# 把当前文件所在文件夹的路径加入到PYTHONPATH
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from service.face_alg.landmark_inference import LandmarkInference
from service.face_alg.detect_inference import DetectInference

# 图像文件夹路径
img_folder_path = "/home/guohao826/AppAlgorithmFace/src/service/face_alg/process_file/imgs"

# 结果图像保存文件夹路径
output_folder = "/home/guohao826/AppAlgorithmFace/src/service/face_alg/results"

# 初始化人脸检测模型和关键点定位模型
detect_model_path = "/home/guohao826/AppAlgorithmFace/models/detection_model/facedetect_640.pth"
landmark_model_path = "/home/guohao826/AppAlgorithmFace/models/detection_model/pfld128_20200211.pth.tar"

detector = DetectInference(model_path=detect_model_path, gpuid=-1, input_size=640, detect_threshold=0.8, nms=0.4)
landmark_detector = LandmarkInference(model_path=landmark_model_path, gpuid=-1)

# 获取图像文件列表
img_files = [f for f in os.listdir(img_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# 遍历图像文件进行检测和关键点定位
for img_file in img_files:
    img_path = os.path.join(img_folder_path, img_file)
    img = cv2.imread(img_path)

    # 人脸检测
    detected_boxes, _ = detector.detect([img])

    # 遍历每个检测到的人脸框，进行关键点定位
    for boxes in detected_boxes:
        for box in boxes:
            # 提取人脸框信息
            x1, y1, x2, y2, score = box[:5]

            # 裁剪人脸区域
            face_img = img[int(y1):int(y2), int(x1):int(x2)]

            # 关键点定位
            landmarks = landmark_detector.inference(face_img, face_size=int(max(x2 - x1, y2 - y1)))

            # 在原图上绘制人脸框和关键点
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            for point in landmarks:
                cv2.circle(img, (int(point[0] + x1), int(point[1] + y1)), 2, (255, 0, 0), -1)

    # 保存结果图像
    result_img_path = os.path.join(output_folder, 'result_' + img_file)
    cv2.imwrite(result_img_path, img)

# 释放图像窗口（如果使用cv2.imshow的话）
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
