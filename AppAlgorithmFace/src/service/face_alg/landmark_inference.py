import numpy as np
import torch
from torchvision import transforms
import cv2

from landmarklib.pfld import PFLDInference


class LandmarkInference:
    def __init__(self, model_path, gpuid=-1, input_size=128):
        self.model_path = model_path

        self.gpuid = gpuid
        self.input_size = input_size
        self.model = PFLDInference()
        if gpuid < 0:
            checkpoint = torch.load(self.model_path,
                                    map_location=lambda storage, loc: storage)
            self.device = torch.device('cpu')
        else:
            checkpoint = torch.load(
                self.model_path,
                map_location=lambda storage, loc: storage.cuda(gpuid))
            self.device = torch.device('cuda')

        self.model.load_state_dict(checkpoint['plfd_backbone'], strict=False)

        self.model.eval()
        self.model = self.model.to(self.device)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def preprocess(self, box, img):
        height, width = img.shape[:2]
        x1, y1, x2, y2 = (box[:4] + 0.5).astype(np.int32)
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        face_size = int(max([w, h]) * 1.1)
        cx = x1 + w // 2
        cy = y1 + h // 2
        x1 = cx - face_size // 2
        x2 = x1 + face_size
        y1 = cy - face_size // 2
        y2 = y1 + face_size

        dx = max(0, -x1)
        dy = max(0, -y1)
        x1_valid = max(0, x1)
        y1_valid = max(0, y1)

        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2_valid = min(width, x2)
        y2_valid = min(height, y2)

        cropped = img[y1_valid:y2_valid, x1_valid:x2_valid]
        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            cropped = cv2.copyMakeBorder(cropped, dy, edy, dx, edx,
                                         cv2.BORDER_CONSTANT, 0)

        cropped = cv2.resize(cropped, (self.input_size, self.input_size))

        input = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        return input, face_size, (x1, y1)

    def inference(self, faceimg, face_size):
        input = self.transform(faceimg).unsqueeze(0)
        input = input.to(self.device)
        print('Input shape: ',input.shape)
        _, landmarks = self.model(input)

        pre_landmark = landmarks[0]
        landmark = pre_landmark.cpu().detach().numpy().reshape(
            -1, 2) * [face_size, face_size]
        return landmark
