import torch
import numpy as np
import cv2
from antispoofinglib.BasicModule import MyresNet34


class AntiSpoofingInference:
    def __init__(self, image_size=224, model_path=None, use_gpu=False, transform=None, data_source=None):
        self.transform = transform
        self.scale = 3
        self.image_size = 224
        self.use_gpu = use_gpu

        self.model = MyresNet34().eval()
        self.model.load(model_path, self.use_gpu)

        if self.use_gpu:
            self.model.cuda()
        self.model.train(False)

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

        return np.array(landmarks_68)

    def landermark_plus_leftcorner(self, input_landmarks, left_corner):
        landmarks_98 = []
        for one in input_landmarks:
            landmarks_98.append(one + left_corner)
        return np.array(landmarks_98)

    def maxbox(self, faces):
        area = []
        faces = np.array(faces)
        for face in faces:
            box = face.astype(np.int)
            area.append((box[2] - box[0]) * (box[3] - box[1]))
        index = np.argmax(np.asarray(area))
        return index

    def det_img(self, detector_model, landmark_model, frame):
        boxes, ldmks = detector_model.detect([frame])
        bounding_boxes = boxes[0]

        if len(bounding_boxes) > 0:
            # landmarks = np.matrix([[p.x, p.y] for p in self.predictor(frame, faces[0]).parts()])
            index = self.maxbox(bounding_boxes)
            box = bounding_boxes[index]

            faceimg, face_size, (x1, y1) = landmark_model.preprocess(box, frame)
            landmark_QC = landmark_model.inference(faceimg, face_size)
            landermark_98 = self.landermark_plus_leftcorner(landmark_QC, np.array([x1, y1]))

            return [np.around(landermark_98)]

    def crop_with_ldmk(self, image, landmark):
        ct_x, std_x = landmark[:, 0].mean(), landmark[:, 0].std()
        ct_y, std_y = landmark[:, 1].mean(), landmark[:, 1].std()

        std_x, std_y = self.scale * std_x, self.scale * std_y

        src = np.float32([(ct_x, ct_y), (ct_x + std_x, ct_y + std_y), (ct_x + std_x, ct_y)])
        dst = np.float32([((self.image_size - 1) / 2.0, (self.image_size - 1) / 2.0),
                          ((self.image_size - 1), (self.image_size - 1)),
                          ((self.image_size - 1), (self.image_size - 1) / 2.0)])
        retval = cv2.getAffineTransform(src, dst)
        result = cv2.warpAffine(image,
                                retval, (self.image_size, self.image_size),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
        return result

    # 第二步装载数据，返回[img,label]
    def get_data(self, image_path):
        img = cv2.imread(image_path)
        return np.transpose(np.array(img, dtype=np.float32), (2, 0, 1)), image_path

    # 第二步装载数据，返回[img,label]
    def get_data_by_frame(self, detector_model, landmark_model, frame):
        img = frame
        # start = cv2.getTickCount()
        ldmk = np.asarray(self.det_img(detector_model, landmark_model, img), dtype=np.float32)
        # time = (cv2.getTickCount()-start)/cv2.getTickFrequency()
        # print ('Time:%.3fs'%time)
        if 0:
            for pred in ldmk:
                for i in range(pred.shape[0]):
                    x, y = pred[i]
                    cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        ldmk = ldmk[np.argsort(np.std(ldmk[:, :, 1], axis=1))[-1]]
        img = self.crop_with_ldmk(img, ldmk)

        image_path = ''
        if 0:
            print(img.shape)
            cv2.imwrite('crop.jpg', img)

        return np.transpose(np.array(img, dtype=np.float32), (2, 0, 1)), image_path

    def __len__(self):
        return len(self.img_label)

    def process_img(self, img_path):
        data, _ = self.get_data(img_path)
        data = data[np.newaxis, :]
        data = torch.FloatTensor(data)
        with torch.no_grad():
            if self.use_gpu:
                data = data.cuda()
            outputs = self.model(data)
            outputs = torch.softmax(outputs, dim=-1)
            preds = outputs.to('cpu').numpy()
            attack_prob = preds[:, 1]
        return attack_prob[0]

    def process_cv_frame(self, detector_model, landmark_model, cv_frame):
        data, _ = self.get_data_by_frame(detector_model, landmark_model, cv_frame)
        data = data[np.newaxis, :]
        data = torch.FloatTensor(data)
        with torch.no_grad():
            if self.use_gpu:
                data = data.cuda()
            outputs = self.model(data)
            outputs = torch.softmax(outputs, dim=-1)
            preds = outputs.to('cpu').numpy()
            attack_prob = preds[:, 1]
        return attack_prob[0]
