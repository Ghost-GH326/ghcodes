#!/usr/bin/env python
# coding=utf-8

import cv2
import sys
import os
import json
import queue
from tqdm import tqdm
import numpy as np
from mylog import logger
from multiprocessing import Manager, Process
from matlab_cp2tform import get_similarity_transform_for_cv2

file_name = os.path.basename(__file__)
logger = logger(file_name)


class AlignProcess(Process):
    def __init__(self, queue, info_queue, num_queue, save_root,
                 align_shape, process_num):
        Process.__init__(self)
        self._queue = queue
        self._info_queue = info_queue
        self._save_root = save_root
        self._num_queue = num_queue
        self._process_num = process_num
        self._align_shape = align_shape

    def run(self):
        while True:
            try:
                tp_dict = self._queue.get(block=False)
                info_dict = align_face(tp_dict, self._align_shape, self._save_root)
                self._num_queue.put(1)
                self._info_queue.put(info_dict)
                processed_num =self._num_queue.qsize()
                if processed_num % 1000 == 0:
                    logger.info('aligend {} faces'.format(processed_num))
            except queue.Empty:
                logger.info('queue empty, {} process quit'.format(self._process_num))
                break
            except Exception as e:
                logger.info(e)


def align_face(ld_dict, align_shape, save_root):
    path = ld_dict['path']
    img = cv2.imread(path)
    label = ld_dict['label']
    points = ld_dict['points']
    ref_pts = [[30.2946, 51.6963], [65.5318, 51.5014],
               [48.0252, 71.7366], [33.5493, 92.3655],
               [62.7299, 92.2041]]
    src_pts = np.array(points).reshape(5, 2)
    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)
    tfm = get_similarity_transform_for_cv2(s, r)
    aligned_img = cv2.warpAffine(img, tfm, (align_shape[0], align_shape[1]),
             flags=cv2.INTER_CUBIC)
    parts = path.split('/')
    im_name = parts[-1]
    sub_root = os.path.join(save_root, str(label))
    new_path = os.path.join(save_root, str(label), im_name)
    mkdir_p(sub_root)
    cv2.imwrite(new_path, aligned_img)
    ld_dict['aliged_path'] = new_path
    return ld_dict


def write_info(info_path, queue, num_queue, all_num):
    while num_queue.qsize() < all_num:
        try:
            info_dict = queue.get()
            with open(info_path, 'a+') as f:
                f.write(json.dumps(info_dict)+'\n')
        except Exception as e:
            logger.info(e)
    logger.info('write info process quit')


def align_paths2queue(path, queue, all_num=10000):
    pbar = tqdm(total=all_num)
    with open(path, 'rb') as f:
        for l in f:
            tp_dict = json.loads(l.decode().strip())
            queue.put(tp_dict)
            pbar.update(1)
    logger.info('load path to queue done')


def mkdir_p(save_path):
    try:
        os.makedirs(save_path, 0o777)
    except Exception as e:
        pass

