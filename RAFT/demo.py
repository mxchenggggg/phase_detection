import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder



DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo, flow_idx, padder):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo_rgb = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo_rgb], axis=0)

    img_path = f'./mastoid_demo_flows/flow_{flow_idx}.png'
    cv2.imwrite(img_path, img_flo[:, :, [2,1,0]])
    
    flo = padder.unpad(flo.transpose(2, 0, 1)).transpose(1,2,0)
    flo = np.round((flo + 20.0) / (2. * 20.0) * 255.)
    flo[flo < 0] = 0
    flo[flo > 255] = 255
    hori_path = f'./mastoid_demo_flows/flow_{flow_idx}_hori.png'
    vert_path = f'./mastoid_demo_flows/flow_{flow_idx}_vect.png'
    cv2.imwrite(hori_path, flo[..., 0])
    cv2.imwrite(vert_path, flo[..., 1])


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        flow_idx = 0
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz(image1, flow_up, flow_idx, padder)
            flow_idx += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
