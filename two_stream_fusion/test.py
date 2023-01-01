from torchvision.models import vgg16
import torch.nn.modules as modules
import torch
import torch.nn as nn
import torch 
from copy import deepcopy

from torchsummary import summary
import cv2
import numpy as np

from PIL import Image

from albumentations import (
    Compose,
    Normalize,
    HorizontalFlip,
    ShiftScaleRotate,
)

from torch.utils.data import Dataset, DataLoader

# pretrained = vgg16(pretrained=True)

# spat_features = deepcopy(pretrained.features)

# temp_features = deepcopy(pretrained.features)
# temp_features[0] = nn.Conv2d(20, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 

# first_conv_weight = pretrained.features[0].weight.data
# first_conv_weight = torch.mean(first_conv_weight, dim=1, keepdim = True)
# first_conv_weight = first_conv_weight.repeat(1, 20, 1, 1)

# temp_features[0].weight.data = first_conv_weight

# spat_features.cuda()
# temp_features.cuda()

# summary(spat_features, (3, 225, 400), 80)
# summary(temp_features, (20, 225, 400), 80)

# img_file = "/home/ubuntu/data/mastoid_frames_15fps/V007/003492.png"
# img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
# print(type(img), img.shape, img.dtype, np.min(img), np.max(img))

# rgb_normalize = Normalize(mean=[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
# img = rgb_normalize(image=img)["image"].transpose(2, 0, 1)
# print(type(img), img.shape, img.dtype, np.min(img), np.max(img))

# flow_u_file = "/home/ubuntu/data/mastoid_optical_flow_15fps/V001/009350_hori.png"
# flow_v_file = "/home/ubuntu/data/mastoid_optical_flow_15fps/V001/009350_vect.png"

# flow_u = cv2.imread(flow_u_file, cv2.IMREAD_GRAYSCALE).astype(np.float32)
# flow_v = cv2.imread(flow_v_file, cv2.IMREAD_GRAYSCALE).astype(np.float32)

# flow_uv = np.stack([flow_u, flow_v]) / 255. * 40. - 20.
# flow_uv = flow_uv.transpose(1, 2, 0)
# from opf_visulize import flow_to_image
# flow_img = flow_to_image(flow_uv)
# cv2.imwrite('./test.png', flow_img)

# flow = np.stack([flow_u, flow_v])
# print(type(flow), flow.shape, flow.dtype, np.min(flow), np.max(flow))


# flow_clip = 20.0
# flow = flow / 255. * (2 * flow_clip) - flow_clip
# print(type(flow), flow.shape, flow.dtype, np.min(flow), np.max(flow))

# indices = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3]
# print(len(indices))
# # np.random.shuffle(indices)
# print(indices)

# grouped = zip(*[iter(indices), iter(indices)])
# grouped = zip(*[iter(indices)] * 2)

# for a, b in grouped:
#     print(a, b)

# a = np.array([[10, 20, 30, 40, 50],
#               [6,  7,  8,  9,  10]])
# b = a[:, [0, 2, 4, 3, 1]]
# print(a)
# print(b)
# b[0][0] = 12
# b[0][1] = 12

# print(a)
# print(b)

# print(np.ceil(8 / 3))


from torchmetrics import Accuracy, Precision, ConfusionMatrix
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt

classes = ["Tegmen", "SS", "EAC", "Open_antrum", "Facial_recess"]
num_classes = 5

target = torch.tensor(np.repeat(np.arange(5), 20))
preds = torch.tensor(np.random.permutation(np.repeat(np.arange(5), 20)))

acc_all_m = Accuracy(num_classes=num_classes)
acc_per_m = Accuracy(num_classes=num_classes, average=None)

prec_all_m = Precision(num_classes=num_classes)
prec_per_m = Precision(num_classes=num_classes, average=None)

c_matrix_m = ConfusionMatrix(num_classes=num_classes)
c_matrix_norm_m = ConfusionMatrix(num_classes=num_classes, normalize='true')

acc_all = acc_all_m(preds, target)
acc_per = acc_per_m(preds, target)
acc = np.append(acc_per, acc_all)

prec_all = prec_all_m(preds, target)
prec_per = prec_per_m(preds, target)
prec = np.append(prec_per, prec_all)

metrics_df = pd.DataFrame(np.stack([acc, prec]), index=['acc','prec'], columns=classes + ['all'])

c_matrix = c_matrix_m(preds, target).numpy()
c_matrix_norm = c_matrix_norm_m(preds, target).numpy()

cm_df = pd.DataFrame(c_matrix, index=classes, columns=classes)
cm_df_norm = pd.DataFrame(c_matrix_norm, index=classes, columns=classes)

plt.figure(figsize=(12, 7))    
fig = sn.heatmap(cm_df_norm, annot=True).get_figure()

plt.savefig("test.png")