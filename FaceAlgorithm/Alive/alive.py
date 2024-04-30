import cv2
import dlib
import json
import numpy as np
from numpy import mat


class LiveFaceDetection:

    BLICK_EYES_ERROR = 1
    OPEN_MOUSE_ERROR = 2
    SHAKE_HEAD_ERROR = 3
    FACE_PREPARE_OK_ERRORS = [1, 2, 3]  # prepare 阶段，此失败列表是正常的
    FACE_HIDE_ERROR = 4
    FACE_CENTER_ERROR = 5
    FACE_FAR_ERROR = 6
    FACE_DIRECT_ERROR = 7
    FACE_LIGHT_ERROR = 8
    ALIVE_SCAN_ERROR = 9  # 扫描所有动作仍然失败
    ANTI_CHECK_ERROR = 10  # 防伪检测失败
    """Fave Huo Ti Detection.

        need put in frame squence ,the give the return yes or no

        Attributes:
            null
        """
    def __init__(self, model_path, orientation, threshold_mouse, threshold_move, threshold_eye, threshold_aligh,
                 win_width, win_height):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_path)

        self.THRESHOLD_ORIENTATION = orientation  # [0, 1], 面部中心到左右眼角的距离比， 越小越严
        self.THRESHOLD_MOUSE = threshold_mouse  # [0, 无穷]， 嘴部面积的变化长宽比，越大越严
        self.THRESHOLD_MOVE = threshold_move  # [0, 无穷]， 20帧算法的输出总和，越小越严
        self.THRESHOLD_EYE = threshold_eye  # [0，1], 眼睛特征点变化的方差的比例， 越大越严
        self.THRESHOLD_ALIGH = threshold_aligh  # 对齐要求的阈值，越小越严
        self.WIN_WIDTH = win_width
        self.WIN_HEIGHT = win_height
        self.EAR_THRESH = 0.20
        self.EAR_CONSEC_FRAMES_MIN = 1
        self.EAR_CONSEC_FRAMES_MAX = 2
        self.C_X = self.WIN_WIDTH / 2
        self.C_Y = self.WIN_HEIGHT / 2
        self.F_X = self.C_X / np.tan(60 / 2 * np.pi / 180)
        self.F_Y = self.F_X

        self.cam_matrix = np.array([self.F_X, 0.0, self.C_X, 0.0, self.F_Y, self.C_Y, 0.0, 0.0,
                                    1.0]).reshape(3, 3).astype(np.float32)

        self.dist_coeffs = np.array(
            [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0,
             -1.3073460323689292e+000]).reshape(5, 1).astype(np.float32)
        self.object_pts = np.float32([[6.825897, 6.760612, 4.402142], [1.330353, 7.122144, 6.903745],
                                      [-1.330353, 7.122144, 6.903745], [-6.825897, 6.760612, 4.402142],
                                      [5.311432, 5.485328, 3.987654], [1.789930, 5.393625, 4.413414],
                                      [-1.789930, 5.393625, 4.413414], [-5.311432, 5.485328, 3.987654],
                                      [2.005628, 1.409845, 6.165652], [-2.005628, 1.409845, 6.165652]])
        self.reprojectsrc = np.float32([[10.0, 10.0, 10.0], [10.0, 10.0, -10.0], [10.0, -10.0, -10.0],
                                        [10.0, -10.0, 10.0], [-10.0, 10.0, 10.0], [-10.0, 10.0, -10.0],
                                        [-10.0, -10.0, -10.0], [-10.0, -10.0, 10.0]])

    def get_head_pose(self, shape):
        image_pts = np.float32([
            shape[17], shape[21], shape[22], shape[26], shape[36], shape[39], shape[42], shape[45], shape[31], shape[35]
        ])

        _, rotation_vec, translation_vec = cv2.solvePnP(self.object_pts, image_pts, self.cam_matrix, self.dist_coeffs)

        reprojectdst, _ = cv2.projectPoints(self.reprojectsrc, rotation_vec, translation_vec, self.cam_matrix,
                                            self.dist_coeffs)

        reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

        # calc euler angle
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

        return reprojectdst, euler_angle

    def isRealFace(self, mid_values):
        biggestClosed = max(mid_values.closedMouse)
        smallestOpened = min(mid_values.openedMouse)
        return (smallestOpened / biggestClosed) > self.THRESHOLD_MOUSE

    def isLightnessEnough(self, frame, face_box, img_w, img_h):
        left_x = int(face_box[0])
        left_y = int(face_box[1])
        right_x = int(face_box[2])
        right_y = int(face_box[3])

        if left_x < 1:
            left_x = 0
        if left_y < 1:
            left_y = 0
        if right_x > (img_w - 1):
            right_x = (img_w - 1)
        if right_y > (img_h - 1):
            right_y = (img_h - 1)

        crop_face = frame[left_y:right_y, left_x:right_x]
        img_hsv = cv2.cvtColor(crop_face, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img_hsv)
        bright_value = np.mean(np.mean(v))

        # value from [0-255] 0 is dark
        if bright_value < 40:
            return False
        return True

    def isBig(self, face_box, img_w, img_h):
        area = (face_box[2] - face_box[0]) * (face_box[3] - face_box[1])
        anchorPer = area / (img_h * img_w)

        if anchorPer > 0.13:
            return True
        else:
            return False

    def index_98to68(self, input_landmarks, left_corner):
        indexs = [
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 33, 34, 35, 36, 37, 42, 43, 44, 45, 46, 51,
            52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 63, 64, 65, 66, 68, 69, 70, 72, 73, 75, 76, 77, 78, 79, 80, 81, 82,
            83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95
        ]
        change_index = [61, 66, 70, 73]
        landmarks_68 = []
        for index in indexs:
            if index in change_index:
                landmark = (input_landmarks[index] + input_landmarks[index + 1]) / 2
                landmarks_68.append(landmark + left_corner)
            else:
                landmarks_68.append(input_landmarks[index] + left_corner)

        return mat(landmarks_68)

    def landermark_plus_leftcorner(self, input_landmarks, left_corner):
        landmarks_98 = []
        for one in input_landmarks:
            landmarks_98.append(one + left_corner)
        return mat(landmarks_98)

    def eye_aspect_ratio_six_points(self, eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def eye_aspect_ratio_eight_points(self, eye):
        a = np.linalg.norm(eye[1] - eye[7])
        b = np.linalg.norm(eye[2] - eye[6])
        c = np.linalg.norm(eye[3] - eye[5])
        d = np.linalg.norm(eye[0] - eye[4])

        ear = (a + b + c) / (3.0 * d)
        return ear

    '''
    def eyeIdentificate(self, detection_success, lanmark_in, mid_values):
        if (detection_success):
            landmarks_eye = lanmark_in[36:42]
            landmarks_right = lanmark_in[42:48]

            left_err = self.eye_aspect_ratio_six_points(landmarks_eye)
            right_err = self.eye_aspect_ratio_six_points(landmarks_right)

            if left_err < self.EAR_THRESH:
                mid_values.blink_counter[0] += 1
                # otherwise, the eye aspect ratio is not below the blink
                # threshold
            else:
                # if the eyes were closed for a sufficient number of
                # then increment the total number of blinks
                if self.EAR_CONSEC_FRAMES_MIN <= mid_values.blink_counter[0] and mid_values.blink_counter[
                        0] <= self.EAR_CONSEC_FRAMES_MAX:
                    mid_values.blink_total[0] += 1
                mid_values.blink_counter[0] = 0

            if right_err < self.EAR_THRESH:
                mid_values.blink_counter[1] += 1
                # otherwise, the eye aspect ratio is not below the blink
                # threshold
            else:
                # if the eyes were closed for a sufficient number of
                # then increment the total number of blinks
                if self.EAR_CONSEC_FRAMES_MIN <= mid_values.blink_counter[1] and mid_values.blink_counter[
                        1] <= self.EAR_CONSEC_FRAMES_MAX:
                    mid_values.blink_total[1] += 1
                mid_values.blink_counter[1] = 0

            if max(mid_values.blink_total) > 0:
                mid_values.blink_counter[0] = 0
                mid_values.blink_counter[1] = 0
                return True
            else:
                return False
    '''

    def eyeIdentificate_eight_points(self, detection_success, lanmark_in, mid_values):
        if (detection_success):
            landmarks_eye = lanmark_in[60:68]
            landmarks_right = lanmark_in[68:76]

            left_err = self.eye_aspect_ratio_eight_points(landmarks_eye)
            right_err = self.eye_aspect_ratio_eight_points(landmarks_right)
            mean_err = (left_err + right_err) / 2.0

            mid_values.left_err.append(left_err)
            mid_values.right_err.append(right_err)
            mid_values.mean_err.append(mean_err)

            min_mean_err = min(mid_values.mean_err)

            if min_mean_err < self.EAR_THRESH:
                # max_mean_err = max(mid_values.mean_err)
                # sub_err = max_mean_err - min_mean_err
                max_left_err = max(mid_values.left_err)
                min_left_err = min(mid_values.left_err)
                sub_left_err = max_left_err - min_left_err

                max_right_err = max(mid_values.right_err)
                min_right_err = min(mid_values.right_err)
                sub_right_err = max_right_err - min_right_err

                if max([sub_left_err, sub_right_err]) > 0.05 and mid_values.onlyOneFace:
                    return True

            return False

    def mouseIdentificate(self, lanmark_in, mid_values):
        isReal = False
        mouseArea = []
        for i in range(48, 60):
            point = lanmark_in[i]
            mouseArea.append([int(point[0, 0]), int(point[0, 1])])
        mouseArea = np.array(mouseArea)
        mouseSize = cv2.contourArea(mouseArea)
        mouseLength = lanmark_in[54][0, 0] - lanmark_in[48][0, 0]
        mouseRatio = mouseSize / (mouseLength * mouseLength)

        # 判断是否为真实人脸
        biggestClosed = max(mid_values.closedMouse)
        smallestOpened = min(mid_values.openedMouse)

        if (mouseRatio < biggestClosed):
            biggestIndex = mid_values.closedMouse.index(biggestClosed)
            mid_values.closedMouse[biggestIndex] = mouseRatio
            mid_values.numofClosedMouses = mid_values.numofClosedMouses + 1

        if (mouseRatio > smallestOpened):
            smallestIndex = mid_values.openedMouse.index(smallestOpened)
            mid_values.openedMouse[smallestIndex] = mouseRatio
            mid_values.numofOpenedMouses = mid_values.numofOpenedMouses + 1

        if (mid_values.numofClosedMouses >= len(mid_values.closedMouse)) and (mid_values.numofOpenedMouses >= len(
                mid_values.openedMouse)):
            isReal = self.isRealFace(mid_values)

        return isReal

    def orientationIdentificate(self, landmark, mid_values):
        point_nose = landmark[33]
        point_left_eye = landmark[36]
        point_right_eye = landmark[45]

        left_temp_x = (point_nose[0, 0] - point_left_eye[0, 0]) * (point_nose[0, 0] - point_left_eye[0, 0])
        left_temp_y = (point_nose[0, 1] - point_left_eye[0, 1]) * (point_nose[0, 1] - point_left_eye[0, 1])
        right_temp_x = (point_nose[0, 0] - point_right_eye[0, 0]) * (point_nose[0, 0] - point_right_eye[0, 0])
        right_temp_y = (point_nose[0, 1] - point_right_eye[0, 1]) * (point_nose[0, 1] - point_right_eye[0, 1])
        oriVal = (left_temp_x + left_temp_y) / (right_temp_x + right_temp_y)

        if oriVal < 0.7:
            mid_values.left_face = True
        if oriVal > 1.3:
            mid_values.right_face = True

        return mid_values.left_face or mid_values.right_face

    def isCenter(self, face_rect, img_w, img_h):
        # get face_center
        left_top = face_rect[0].left()
        left_top_y = face_rect[0].top()
        right_bottom = face_rect[0].right()
        right_bottom_y = face_rect[0].bottom()

        center_x = (left_top + right_bottom) / 2
        center_y = (left_top_y + right_bottom_y) / 2
        f_center_x = abs(center_x / (img_w * 0.5) - 1)
        f_center_y = abs(center_y / (img_h * 0.5) - 1)

        face_align_center = (f_center_x < 0.2) and (f_center_y < 0.3)

        return face_align_center

    def isCenter_nose(self, lanmark_in, img_w, img_h):
        p_nose = lanmark_in[30]
        p_x = p_nose[0, 0]
        p_y = p_nose[0, 1]

        border_left = img_w / 4
        border_right = img_w * 3 / 4

        border_up = img_h / 3
        border_down = img_h * 3 / 4

        face_align_center = (p_x > border_left)
        face_align_center = face_align_center and (p_x < border_right)
        face_align_center = face_align_center and (p_y > border_up)
        face_align_center = face_align_center and (p_y < border_down)

        return face_align_center

    def maxbox(self, faces):
        area = []
        faces = np.array(faces)
        for face in faces:
            box = face.astype(np.int)
            area.append((box[2] - box[0]) * (box[3] - box[1]))
        index = np.argmax(np.asarray(area))
        return index

    def run_frame(self,
                  frame,
                  face_detector,
                  lander_detector,
                  iden_type,
                  req_type,
                  mid_values,
                  risk_score="",
                  is_last_img=False):
        """Run one frame

            Args:
                frame: input single frame (size,280*280)
                iden_type: detection type, 1 for eyes, 2 for mouse.

            Returns:
                detect_frame: 进行人脸识别计算的 frame
                error_msg: 错误消息
                error_type: 错误类型
            """
        error_msg = ""
        error_type = 0
        dim = frame.shape
        img_h = dim[0]
        img_w = dim[1]

        array_lst = []
        detect_frame = frame
        # faces = self.detector(frame, 0)
        array_lst.append(frame)
        boxes, ldmks = face_detector.detect(array_lst)
        bounding_boxes = boxes[0]

        # if len(bounding_boxes) > 1:
        # mid_values.onlyOneFace = False

        # if len(faces) == 1:
        if len(bounding_boxes) > 0:
            # landmarks = np.matrix([[p.x, p.y] for p in self.predictor(frame, faces[0]).parts()])
            index = self.maxbox(bounding_boxes)
            box = bounding_boxes[index]

            # is Lightness enough
            isLightFace = self.isLightnessEnough(frame, box, img_w, img_h)

            # get face_size
            isBigFace = self.isBig(box, img_w, img_h)

            # score = box[4]
            faceimg, face_size, (x1, y1) = lander_detector.preprocess(box, frame)
            landmark_QC = lander_detector.inference(faceimg, face_size)
            landermark_98 = self.landermark_plus_leftcorner(landmark_QC, np.array([x1, y1]))
            landmarks = self.index_98to68(landmark_QC, np.array([x1, y1]))

            # get face_center
            # face_align_center = self.isCenter(faces, img_w, img_h)
            face_align_center = self.isCenter_nose(landmarks, img_w, img_h)
            if req_type == 1:
                face_align_center = 1

            # get the euler_angle
            reprojectdst, euler_angle = self.get_head_pose(landmarks)

            aligned = 0
            aligned_sum = abs(euler_angle[0, 0]) + abs(euler_angle[1, 0]) + abs(euler_angle[2, 0])
            aligned = aligned_sum < self.THRESHOLD_ALIGH
            if isLightFace:
                if face_align_center:
                    if iden_type == 999:
                        # select better img to recognition
                        if aligned_sum < mid_values.min_align:
                            mid_values.min_align = aligned_sum
                            mid_values.better_frame = detect_frame
                        # 最后一帧且低风险用户
                        auto_pass = is_last_img and (risk_score == "-1" or risk_score == "0")
                        # 扫描所有动作
                        if auto_pass or self.orientationIdentificate(
                                landmarks, mid_values) or self.eyeIdentificate_eight_points(
                                    1, landermark_98, mid_values) or self.mouseIdentificate(landmarks, mid_values):
                            detect_frame = mid_values.better_frame if mid_values.better_frame is not None else detect_frame
                            return detect_frame, error_msg, error_type
                        else:
                            error_msg = "Alive check failed!"
                            error_type = self.ALIVE_SCAN_ERROR
                            return detect_frame, error_msg, error_type
                    if isBigFace:
                        if iden_type == 3:
                            if self.orientationIdentificate(landmarks, mid_values):
                                return detect_frame, error_msg, error_type
                            else:
                                error_msg = "Shake the head!"
                                error_type = self.SHAKE_HEAD_ERROR
                        if aligned:
                            if iden_type == 1:
                                # if self.eyeIdentificate(1, landmarks, mid_values):
                                if self.eyeIdentificate_eight_points(1, landermark_98, mid_values):
                                    return detect_frame, error_msg, error_type
                                else:
                                    error_msg = "Blick the eyes!"
                                    error_type = self.BLICK_EYES_ERROR
                            if iden_type == 2:
                                if self.mouseIdentificate(landmarks, mid_values):
                                    return detect_frame, error_type, error_type
                                else:
                                    error_msg = "Open your mouse!"
                                    error_type = self.OPEN_MOUSE_ERROR

                        else:
                            error_msg = "Please face directly to the screen!"
                            error_type = self.FACE_DIRECT_ERROR
                    else:
                        error_msg = "Move closer to the screen"
                        error_type = self.FACE_FAR_ERROR
                else:
                    error_msg = "Move head to the center!"
                    error_type = self.FACE_CENTER_ERROR
            else:
                error_msg = "Lightness is not enough"
                error_type = self.FACE_LIGHT_ERROR
        else:
            error_msg = "Don't hide your face"
            error_type = self.FACE_HIDE_ERROR
        return detect_frame, error_msg, error_type


class MidValues:
    '''
    记录单次识别过程中的中间值，避免并发问题
    '''
    def __init__(self, **kwargs):
        self.numofClosedMouses = json.loads(kwargs["numofClosedMouses"]) if kwargs.get("numofClosedMouses") else 0
        self.numofOpenedMouses = json.loads(kwargs["numofOpenedMouses"]) if kwargs.get("numofOpenedMouses") else 0
        # self.blink_counter = json.loads(kwargs["blink_counter"]) if kwargs.get("blink_counter") else [0, 0]
        # self.blink_total = json.loads(kwargs["blink_total"]) if kwargs.get("blink_total") else [0, 0]
        self.closedMouse = json.loads(kwargs["closedMouse"]) if kwargs.get("closedMouse") else [99999, 99999]
        self.openedMouse = json.loads(kwargs["openedMouse"]) if kwargs.get("openedMouse") else [-99999, -99999]
        self.left_face = json.loads(kwargs["left_face"]) if kwargs.get("left_face") else False
        self.right_face = json.loads(kwargs["right_face"]) if kwargs.get("right_face") else False
        self.left_err = []
        self.right_err = []
        self.mean_err = []
        self.onlyOneFace = True
        self.better_frame = None
        self.min_align = 99999


def check_alive(processor, face_detector, landermark_detector, images, action_type, request_type, risk_score, **kwargs):
    '''
    进行活体检测，检测成功则返回可用于识别的图片，否则返回 None 和错误消息。同时返回中间数据
    images, cv2 image 对象
    action_type，动作类型，1 表示 眨眼，2 表示张嘴, 3 表示摇头
    目前只支持单个动作类型
    request_type  0=prepare, 1=alive
    risk_score 风控置信分，-1 0 1 2 3，string 类型，值越低，越安全
    '''
    result_frame = None
    mid_values = MidValues(**kwargs)
    error_msg = "no images"
    error_type = -1

    if request_type == 0:
        frame = images[0]
        img_w = frame.shape[1]
        img_h = frame.shape[0]
        min_edge = min(img_w, img_h)
        if img_w < img_h:
            frame = frame[0:min_edge, 0:min_edge]

        dim = frame.shape
        if dim[0] != processor.WIN_WIDTH or dim[1] != processor.WIN_HEIGHT:
            frame = cv2.resize(frame, (processor.WIN_WIDTH, processor.WIN_HEIGHT))

        detect_frame, error_msg, error_type = processor.run_frame(frame, face_detector, landermark_detector,
                                                                  action_type, request_type, mid_values)
        if error_type == 0:
            return detect_frame, error_msg, error_type, mid_values

    # alive request
    if request_type == 1:
        images_num = len(images)
        frame_index = 0

        for frame in images:
            frame_index += 1
            img_w = frame.shape[1]
            img_h = frame.shape[0]
            if img_w > img_h:
                frame = np.rot90(frame)

            dim = frame.shape
            resize_w = processor.WIN_WIDTH
            resize_h = int(resize_w * dim[0] / dim[1])

            frame = cv2.resize(frame, (resize_w, resize_h))
            detect_frame, error_msg, error_type = processor.run_frame(frame, face_detector, landermark_detector,
                                                                      action_type, request_type, mid_values, risk_score,
                                                                      frame_index == images_num)
            if error_type == 0:
                return detect_frame, error_msg, error_type, mid_values

    return result_frame, error_msg, error_type, mid_values
