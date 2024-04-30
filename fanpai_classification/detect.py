import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from sklearn import decomposition
from sklearn import manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

import copy
from collections import namedtuple
import os, subprocess
import random
import shutil
import time

from model_beta import ResNet, resnet50_config
from tools.train_utils import LRFinder
from tools.runner import train, evaluate, epoch_time
from tqdm import tqdm
from tools.test_utils import get_predictions, plot_confusion_matrix, plot_most_incorrect
from PIL import Image
from tools.detect_utils import check_uniform, detect_uniform_regions

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
ROOT = 'data'
data_dir = os.path.join(ROOT, 'fanpai_data_v2.0')
detect_dir = os.path.join(data_dir, 'detect')
classes = ['improper', 'proper']
OUTPUT_DIM = len(classes)
model = ResNet(resnet50_config, OUTPUT_DIM)
model.load_state_dict(torch.load('tut5-model.pt'))
model.to(device)

# 加载数据
pretrained_size = 224
pretrained_means = [0.485, 0.456, 0.406]
pretrained_stds= [0.229, 0.224, 0.225]

test_transforms = transforms.Compose([
                           transforms.Resize(pretrained_size),
                           transforms.CenterCrop(pretrained_size),
                           transforms.ToTensor(),
                           transforms.Normalize(mean = pretrained_means, 
                                                std = pretrained_stds)
                       ])

imagesTodetect = os.listdir(detect_dir)
for img in tqdm(imagesTodetect):
    imgPath = os.path.join(detect_dir, img)
    # with open(imgPath, "rb") as f:
    #     imgByte = Image.open(f)
    #     imgByte = imgByte.convert("RGB")
    imgByte=Image.open(imgPath)
    img_tensor = test_transforms(imgByte)
    img_tensor = img_tensor.unsqueeze(0) # Add batch dimension

    with torch.no_grad():
        model.eval()
        img_tensor = img_tensor.to(device)
        y_pred, _ = model(img_tensor)
        y_prob = F.softmax(y_pred, dim = -1)
        # print(y_prob)
        predicted_idx = torch.argmax(y_prob, 1)
        print(f'{img} is {classes[predicted_idx]}')
        # 筛选
        # if predicted_idx == 0:
        #     command_1 = f'cp {imgPath} /home/guohao826/fanpai_classification/data/fanpai_data_v2.0/except_images/preImpro_turePro'
        #     subprocess.run(command_1, shell=True)
        # if predicted_idx == 1:
        #     command_2 = f'cp {imgPath} /home/guohao826/fanpai_classification/data/fanpai_data_v2.0/except_images/prePro_tureImpro'
        #     subprocess.run(command_2,shell=True)
