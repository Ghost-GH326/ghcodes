# encoding:utf-8
# align face with landmark

import os
import cv2
import queue
import json
import numpy as np
from mylog import logger
from tqdm import tqdm
from itertools import chain, cycle
import multiprocessing as mp
from multiprocessing import Process

from detect_inference import DetectInference
from matlab_cp2tform import get_similarity_transform_for_cv2

file_name = os.path.basename(__file__)
logger = logger(file_name)


def get_gpu_queue(gpus, process_num):
    gpu_ids = mp.Queue()
    if len(gpus) > 0:
        gpu_id_cycle_iterator = cycle(gpus)
    for i in range(process_num):
        gpu_ids.put(next(gpu_id_cycle_iterator))
    return gpu_ids


def detect_imgs(imgs_lst):
    detectinfer.inference(imgs_lst)


def init_detectinfer(gpus, ld_queue, num_queue, model_path, inputsize, network,
                     threshold):
    global detectinfer
    if not gpus.empty():
        gpu_id = gpus.get()
    detectinfer = DetectInfer(ld_queue, num_queue, model_path, inputsize,
                              gpu_id, network, threshold)
    return detectinfer


class DetectInfer:
    def __init__(self, ld_queue, num_queue, model_path, inputsize, gpu_id,
                 network, threshold):
        Process.__init__(self)
        self._ld_queue = ld_queue
        self._num_queue = num_queue
        self._detectmodel = self._init_detectmodel(model_path, inputsize,
                                                   gpu_id, network, threshold)

    def _init_detectmodel(self, model_path, inputsize, gpu_id, network,
                          threshold):
        retinamodel = DetectInference(model_path, gpu_id, inputsize, threshold)
        return retinamodel

    def _read_imgs(self, img_paths):
        imgs_lst = []
        path_lst = []
        for path_i in img_paths:
            img = cv2.imread(path_i)
            try:
                img.shape
                imgs_lst.append(img)
                path_lst.append(path_i)
            except Exception as e:
                pass
            self._num_queue.put(1)
            processed_num = self._num_queue.qsize()
            if processed_num % 100 == 0:
                logger.info('processing {}'.format(processed_num))
        return imgs_lst, path_lst

    def inference(self, paths_labels):
        try:
            img_paths, labels = paths_labels
            batch_imgs, img_paths = self._read_imgs(img_paths)
            batch_len = len(img_paths)
            faces, ldmks = self._detectmodel.detect(batch_imgs)
            for idx in range(batch_len):
                box = faces[idx][0].tolist()
                ldmk = ldmks[idx][0].tolist()
                ldmk = list(chain.from_iterable(ldmk))
                path = img_paths[idx]
                label = labels[idx]
                tp_dict = {
                    'path': path,
                    'points': ldmk,
                    'box': box,
                    'label': label
                }
                self._ld_queue.put(tp_dict)
        except Exception as e:
            logger.info(e)


def detect_path2queue(path, path_lst, batch_size, all_num):
    pbar = tqdm(total=all_num)
    batch_paths = []
    batch_labels = []
    with open(path, 'rb') as f:
        for l in f:
            tp_dict = json.loads(l.decode().strip())
            path = tp_dict['path']
            label = tp_dict['label']
            if len(batch_labels) < batch_size:
                batch_paths.append(path)
                batch_labels.append(label)
            else:
                path_lst.append([batch_paths, batch_labels])
                batch_paths, batch_labels = [path], [label]
            pbar.update(1)
    path_lst.append([batch_paths, batch_labels])
    logger.info('load path to queue done')
    return path_lst
