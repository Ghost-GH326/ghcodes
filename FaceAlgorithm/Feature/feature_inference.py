# encoding: utf-8
# pytorch face feature extractor

import logging
import torch
import sys
import cv2
import os
import numpy as np
from torchvision import transforms
from itertools import chain


from ..Detect.detect_inference import DetectInference
from importlib import import_module
from jdconfig import Config
from matlab_cp2tform import get_similarity_transform_for_cv2

filename = os.path.basename(__file__)


class PthExtractor:
    """
    face feature extract class for pytorch model

    """
    def __init__(self,
                 model_path,
                 config_path,
                 device_id,
                 batch_size=16,
                 im_shape=[112, 112]):
        """
        model inintialization
        Args:
            model_path: string, the path of pth model
            config_path: string, the path of training config
            device_id: int, gpu_id.If machine has 4 gpus, the possible values are 0,1,2,3
            batch_size: the number of images inferenceed every time.The number
            should be appropriate, if it is too large, the error will be 'out of Memory error'.
            im_shape: the model input image size
        """
        if device_id >= 0:
            self.device = torch.device('cuda:{}'.format(device_id))
        else:
            self.device = torch.device('cpu')
        self.trans = self._get_trans()
        self.batch_size = batch_size
        self.model = self._get_model(model_path, config_path)
        self.im_tensor = torch.zeros([batch_size, 3, im_shape[0], im_shape[1]],
                                     dtype=torch.float32,
                                     device=self.device)

    def _get_network(self, model_path, conf_path):
        model_root = os.path.dirname(model_path)
        sys.path.append(model_root)
        filename = os.path.basename(conf_path)
        conf = Config(conf_path)
        conf.parse_refs()
        conf = conf(**{
            'meta.path': conf_path,
            'meta.filename': filename,
        })
        conf.network.kwargs.extract_feat = True
        get = self._import_lib('%s.get_network' % conf.network.submodule)
        net = get(conf.network)
        return net

    def _import_lib(self, dot_path):
        tokens = dot_path.split('.')
        module = import_module('.'.join(tokens[:-1]))
        method = getattr(module, tokens[-1])
        return method

    def _get_model(self, model_path, config_path):
        model = self._get_network(model_path, config_path)
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict['network'], strict=False)
        model.eval()
        model.to(self.device)
        return model

    def _get_trans(self):
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        return trans

    def _data_transform(self, imarray):
        im = cv2.cvtColor(imarray, cv2.COLOR_BGR2RGB)
        trans_im = self.trans(imarray)
        return trans_im

    def inference(self, key_imarray_dict, norm=True):
        """
        get face features by batch inference

        Args:
            key_imarray: dict type, eg: {key1: imarray1, key2:imarray2}
            imarray is RGB format images array and the width and height should be
            same as im_shape
            norm: type bool, default is True.

        Return:
            path_feat: dict type, eg:{key1: feat_ls1, key2: feat_ls2}
            feat_ls is feature list
        """
        length = len(key_imarray_dict)
        keys = list(key_imarray_dict.keys())
        path_feat = {}
        with torch.no_grad():
            for start_idx in range(0, length, self.batch_size):
                end_idx = min(length, start_idx + self.batch_size)
                batch_len = end_idx - start_idx
                for idx in range(0, batch_len):
                    i = start_idx + idx
                    key = keys[i]
                    imarray = key_imarray_dict[key]
                    im = self._data_transform(imarray)
                    self.im_tensor[idx] = im.to(self.device)
                features = self.model(self.im_tensor).cpu().numpy()
                for idx in range(0, batch_len):
                    i = start_idx + idx
                    key = keys[i]
                    feat = features[idx]
                    if norm:
                        feat = feat / np.linalg.norm(feat)
                    feat = feat.tolist()
                    path_feat[key] = feat
        return path_feat


def align_face(img, points, align_shape):
    """
    face alignmemt with original images and ladnmarks

    Args:
        img: image array, cv2 format with RGB chanel order.
        points: a list with ten float number which means landmark position in img.
                the order is as follows:
                [left_eye_x, left_eye_y, right_eye_x, right_eye_y, 
                nose_tip_x, nose_tip_y, mouth_left_x, mouth_left_y,
                mouth_right_x, mouth_right_y]
        align shape: aligned images shape, first number is width, second is height

    Return:
        aligned_img: aligened image array
    """
    mean_points = [0.31556875, 0.4615741071428571,
                   0.6826229166666667, 0.4598339285714285,
                   0.5002624999999999, 0.6405053571428571,
                   0.34947187500000004, 0.8246919642857142,
                   0.6534364583333333, 0.8232508928571428]
    from_points = np.asarray(points, dtype='float').reshape(5, 2)
    to_points = np.asarray(mean_points, dtype='float').reshape(5, 2)
    to_points[:, 0] = to_points[:, 0] * align_shape[0]
    to_points[:, 1] = to_points[:, 1] * align_shape[1]
    tfm = get_similarity_transform_for_cv2(from_points, to_points)
    aligned_img = cv2.warpAffine(img, tfm, (align_shape[0], align_shape[1]),
                  flags=cv2.INTER_CUBIC)
    return aligned_img


def maxbox(faces):
    area = []
    faces = np.array(faces)
    for face in faces:
        box = face.astype(np.int)
        area.append((box[2] - box[0])*(box[3] - box[1]))
    index = np.argmax(np.asarray(area))
    return index


def detect_align(detector, key_imarray, align_shape=[112, 112]):
    """
    get landmarks and face positions

    Args:
        detector: retinaface object: initialized detection model
        key_imarray: dict type, {key1: imarray1, key2: imarray2},
        imarray is images array with BGR order
        align_shape: aligned images shape, first number is width, second is height

    Return:
        res_dict: dict type, {key1: face_imarray1, key2: face_imarray2)}.
        face_imarray is aligend face image np.array with BGR order
    """
    array_lst = []
    key_lst = []
    batch_len = len(key_imarray)
    for key in key_imarray:
        key_lst.append(key)
        array_lst.append(key_imarray[key])
    boxes, ldmks = detector.detect(array_lst)
    res_dict = {}
    for idx in range(batch_len):
        key = key_lst[idx]
        maxid = maxbox(boxes[idx])
        res_dict[key] = {}
        box = boxes[idx][maxid].tolist()
        imarray = array_lst[idx]
        ldmk = ldmks[idx][maxid].tolist()
        # ldmk = list(chain.from_iterable(ldmk))
        aligned_img = align_face(imarray, ldmk, align_shape)
        res_dict[key] = aligned_img
    return res_dict


def get_im_array(path1):
    im = cv2.imread(path1)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def get_sim(f1, f2):
    f1 = np.array(f1)
    f2 = np.array(f2)
    return np.dot(f1, f2)


def test_feature_unit(conf, key_array):
    # init model
    model_path = conf.FEATURE_MODEL_PATH
    config_path = conf.MODEL_CONFIG
    device = conf.GPUID
    feature_model = PthExtractor(model_path,
                                 config_path,
                                 device,
                                 batch_size=conf.BATCH_SIZE,
                                 im_shape=conf.INPUT_SHAPE)
    path_feat = feature_model.inference(key_array)
    # compute similarity socre
    # positive pairs
    path1, path2, path3 = conf.ALIGN_PATHS
    ERROR = conf.ERROR
    compute_score = get_sim(path_feat[path1], path_feat[path2])
    print(compute_score)
    err1 = abs(conf.POSITIVE_SCORE - compute_score)
    if err1 < ERROR:
        print(
            'positive pair score is right, and error is {}'.format(err1))
    else:
        print('similarity is not expected, please check')
    # negative pairs
    compute_score = get_sim(path_feat[path1], path_feat[path3])
    print(compute_score)
    err2 = abs(conf.NEGATIVE_SCORE - compute_score)
    if err2 < ERROR:
        print(
            'negative pair score is right, and error is {}'.format(err2))
    else:
        print('similarity is not expected, please check')

    if err1 < ERROR and err2 < ERROR:
        print('[===============feature unit is ok ================]')
        return True


def test_align_unit(conf):
    # init detector
    model_path = conf.DETECT_MODEL_PATH
    inputsize = conf.INPUT_SHAPE
    network = conf.NETNAME
    threshold = conf.THRESHOLD
    gpu_id = conf.GPUID
    detector = DetectInference(model_path, gpu_id, inputsize, threshold)
    # align
    key_imarray = {}
    imgs_paths = conf.ALIGN_PATHS
    for path in imgs_paths:
        im = cv2.imread(path)
        key_imarray[path] = im
    res_dict = detect_align(detector,
                            key_imarray,
                            align_shape=conf.ALIGN_SHAPE)

    aligned_im = res_dict[conf.ALIGN_PATHS[0]]
    print(aligned_im)
    # save aligned face
    test_align_im = cv2.imread(conf.TEST_ALIGNED_IMG)
    err_sum = np.sum(np.where(abs(aligned_im - test_align_im) > conf.ERROR))
    if err_sum < 1000000000:
        print('[===============align unit is ok===================]')
        return res_dict
    else:
        print(
            'alignment is not right, please check, error sum is {}'.format(
                err_sum))

