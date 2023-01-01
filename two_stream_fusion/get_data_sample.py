import numpy as np
import cv2
import os

from opf_visulize import flow_to_image
from mastoid_dataset import MastoidTwoSteamDataset, BatchSampler, MastoidTransform
from torch.utils.data import DataLoader

def get_data_sample():

    train_transform = MastoidTransform(
        augment=True, hflip_p=0.5, affine_p=0.5, rotate_angle=0.,
        scale_range=(0.9, 1.1),
        color_jitter_p=0.5, brightness=0.2, contrast=0.2,
        saturation=0.2, hue=0.1)

    dataset = MastoidTwoSteamDataset(
        "train", train_transform, 15, [1, 2, 4], 5, 10, "class", "Task")
    dataloader = DataLoader(dataset,
                            batch_sampler=BatchSampler(
                                dataset.group_idxes, 1, debug=True))

    rgb, flow, label_one_hot = next(iter(dataloader))
    rgb = rgb.squeeze(0)
    flow = flow.squeeze(0)
    label_one_hot = label_one_hot.squeeze(0)

    classes = ["Tegmen", "SS", "EAC", "Open_antrum", "Facial_recess"]
    class_idx = np.argmax(label_one_hot.numpy())
    print(f'data sample {classes[class_idx]}')

    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

    data_sample_path = './data_sample'
    for rgb_idx, rgb_img in enumerate(rgb):
        rgb_img = rgb_img.numpy()
        rgb_img = ((rgb_img * std + mean) * 255.).astype(np.uint8)
        rgb_img = rgb_img.transpose(1, 2, 0)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        rgb_img_path = os.path.join(data_sample_path, f'rgb_img_{rgb_idx}.png')
        cv2.imwrite(rgb_img_path, rgb_img)

    for flow_idx, flow_stk in enumerate(flow):
        flow_stk = flow_stk.numpy()
        for i in range(0, 20, 2):
            flow_uv = flow_stk[i:i + 2, :, :].transpose(1, 2, 0)
            flow_img = flow_to_image(flow_uv)
            flow_img_path = os.path.join(data_sample_path, f'flow_img_{flow_idx:02d}_{i:02d}.png')
            cv2.imwrite(flow_img_path, flow_img)

if __name__ == '__main__':
    get_data_sample()