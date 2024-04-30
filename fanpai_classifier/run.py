"""
coding:UTF-8
@Software:PyCharm
@Author:ZWY
"""
import os
import onnxruntime
import numpy as np
from scipy.special import softmax
try:
    from .downfile import oss_down_bos, oss_upload
except ImportError:
    from downfile import oss_down_bos, oss_upload

from ast import literal_eval
from torchvision.transforms import transforms
from PIL import Image


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        # padding
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w + w_padding, h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h + h_padding))

        img = img.resize(self.size, self.interpolation)

        return img


class ClassifyFilter:

    def __init__(self, cfg):
        """

        Args:
            cfg:
            {
            'detector_threshold': 0.5,
           }
        """
        # 初始化
        model_path = os.path.dirname(__file__) + "/models/rephoto_v3_fp16.onnx"
        session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
        self.input_names = list(map(lambda x: x.name, session.get_inputs()))
        self.output_names = list(map(lambda x: x.name, session.get_outputs()))
        self.session = session

        self.classify_threshold = cfg.get('classify_threshold', 0.5)
        # 预处理参数
        self.pixel_mean = [0.483, 0.461, 0.435]
        self.pixel_std = [0.263, 0.251, 0.269]
        self.resolution = (512, 512)

        trans = [transforms.ToPILImage(),
                 Resize(size=self.resolution),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=self.pixel_mean, std=self.pixel_std)]
        self.trans = transforms.Compose(trans)

    def preprocess(self, img):
        """
        :param img: 原图
        :return: 预处理后的图像
        """
        workimg = self.trans(img)
        return workimg.unsqueeze(dim=0).numpy()

    def limit_class(self, scores):
        """
        过滤不满足阈值的故障类
        Args:
            scores:
        Returns:

        """
        cls = int(np.argmax(scores))
        # 故障类
        if (cls != 0) and (scores[cls] < self.classify_threshold):
            cls = 0
        return cls, float(scores[cls])

    def predict(self, params):
        """
        :param params
        {
            "urls": [
                {
                "index": xxx,
                "url": xxx,
                }
            ]
        }
        :return:

            'idx':{
                pred: [{'idx':idx, 'cls': cls, 'score': score, ...]
                is_rephoto: xxx
            }
        """
        img_input = params['urls']

        # 解析url下载图片
        imgs_indexes = oss_down_bos(img_input)
        results = {}

        is_rephoto = 0
        pred_res = []
        max_score = 0
        max_index = -1

        # 历遍图片
        for img_index in imgs_indexes:
            img = img_index['im']
            idx = img_index['index']

            # 预处理
            workimg = self.preprocess(img)
            pred = self.session.run(self.output_names, {self.input_names[0]: workimg})
            pred = np.squeeze(pred)
            scores = softmax(pred, axis=-1)
            print(scores)
            cls, score = self.limit_class(scores)

            if int(cls) == 1:
                is_rephoto = 1
                if  score > max_score:
                    max_score = score
                    max_index = idx

            pred_res.append({'idx': idx, 'cls': cls, 'score': score})

        results['pred'] = pred_res
        results['is_rephoto'] = is_rephoto
        results['max_score'] = max_score
        results['max_index'] = max_index

        return {"result": results}


def init(cfg):
    return ClassifyFilter(cfg)
