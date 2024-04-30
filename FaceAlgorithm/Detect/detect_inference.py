import torch
import numpy as np
import cv2
from detectlib.layers.functions.prior_box import PriorBox
from detectlib.utils.nms.py_cpu_nms import py_cpu_nms
from detectlib.models.retinaface import RetinaFace
from detectlib.utils.box_utils import decode, decode_landm
import traceback
from ..utils.oss_utils import oss_down

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


class DetectInference:
    '''
    model_path: 模型文件路径
    input_size: 输入图片尺寸
    gpuid: 运行设备
    '''
    def __init__(self, model_path, gpuid=-1, input_size=640, detect_threshold=0.8, nms=0.4):
        self.model_path = model_path
        self.input_size = input_size
        self.max_size = 960
        self.gpuid = gpuid
        self.confidence_threshold = detect_threshold
        self.nms_threshold = nms
        self.model = RetinaFace()
        if gpuid < 0:
            state_dict = torch.load(self.model_path, map_location=lambda storage, loc: storage)
            self.device = torch.device('cpu')
        else:
            state_dict = torch.load(self.model_path, map_location=lambda storage, loc: storage.cuda(gpuid))
            self.device = torch.device('cuda:{}'.format(gpuid))

        self.model.load_state_dict(state_dict, strict=False)

        self.model.eval()
        self.model = self.model.to(self.device)

        priorbox = PriorBox(image_size=(self.input_size, self.input_size))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        self.prior_data = priors.data

        input_shape = torch.Tensor([self.input_size, self.input_size, self.input_size, self.input_size])
        self.input_shape = input_shape.to(self.device)

        landmarks_scale = torch.Tensor([
            self.input_size, self.input_size, self.input_size, self.input_size, self.input_size, self.input_size,
            self.input_size, self.input_size, self.input_size, self.input_size
        ])
        self.landmarks_scale = landmarks_scale.to(self.device)

    def preprocess(self, img):
        img = np.float32(img)
        im_shape = img.shape
        # im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        scale = float(self.input_size) / float(im_size_max)

        if scale != 1:
            im = cv2.resize(img, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        else:
            im = img.copy()

        _im = np.zeros((self.input_size, self.input_size, 3), dtype=np.float32)
        _im[0:im.shape[0], 0:im.shape[1], :] = im

        _im = torch.from_numpy(_im)
        _im = _im.to(self.device)
        _im = _im.permute(2, 0, 1)

        return _im, scale

    '''
    img: 输入图片 batchsize*h*w*c
    return:
        dets: batchsize*face_num*5
                [
                     [
                        [x1,y1,x2,y2,score],
                            ........
                     ],
                     [
                        [x1,y1,x2,y2,score],
                            .......
                     ]
                ]
        landms:batchsize*face_num*10
                [
                    [
                        [x0,y0,......,x4,y4],
                            .......
                    ],
                    [
                        [x0,y0,......,x4,y4],
                            ......
                    ]
                ]


    '''

    def detect(self, ulr, guidetype):
        imgs = oss_down(ulr)
        output_boxes = []
        output_landmarks = []

        im_tensor = torch.zeros([len(imgs), 3, self.input_size, self.input_size],
                                dtype=torch.float32,
                                device=self.device)

        scales = []
        for index in range(len(imgs)):
            im = imgs[index]
            im, scale = self.preprocess(im)
            im_tensor[index] = im
            scales.append(scale)
        batch_loc, batch_conf, batch_landms = self.model(im_tensor)  # forward pass

        for index in range(len(imgs)):
            boxes = decode(batch_loc[index].data, self.prior_data, [0.1, 0.2])
            boxes = boxes * self.input_shape / scales[index]
            boxes = boxes.cpu().numpy()
            scores = batch_conf[index].data.cpu().numpy()[:, 1]
            landms = decode_landm(batch_landms[index].data, self.prior_data, [0.1, 0.2])

            landms = landms * self.landmarks_scale / scales[index]
            landms = landms.cpu().numpy()

            # ignore low scores
            inds = np.where(scores > self.confidence_threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1]
            # order = scores.argsort()[::-1][:args.top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]
            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, self.nms_threshold)

            dets = dets[keep, :]
            landms = landms[keep]
            output_boxes.append(dets)
            output_landmarks.append(landms)
        return output_boxes, output_landmarks

    def predict(self, params):
        try:
            res = self.detect(params["url"], params['guidetype'])
            return res
        except Exception as e:
            return {'respcode': '1', 'msg': traceback.format_exc(), 'msg2': ' '}