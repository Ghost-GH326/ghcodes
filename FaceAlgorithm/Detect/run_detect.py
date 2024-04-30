# -*- coding: utf-8 -*-
import os
import logging
logging.getLogger().setLevel(logging.DEBUG)
logging.debug("PATHH:")
logging.debug(os.getcwd())

from detect_inference import DetectInference


# 初始化返回的是模型
def init(args):
    cur_path = os.path.dirname(__file__)
    args_params = {
    # 模型的参数
    'DETECT_MODEL_PATH':cur_path + '/models/detection_model/facedetect_640.pth',
    'DETECT_GPUID':0,
    'DETECT_INPUT_SHAPE':640,
    'DETECT_THRESHOLD':0.8, 
    }

    model = DetectInference(args_params)
    return model
