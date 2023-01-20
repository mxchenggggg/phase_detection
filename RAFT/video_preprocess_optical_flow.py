import sys
sys.path.append('core')

import pandas as pd
import os
import glob
import cv2
from tqdm import tqdm
import numpy as np
import torch

import argparse

from core.raft import RAFT
from core.utils.utils import InputPadder

DEVICE = 'cuda'

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--frame_stride', type=int, required=True)
parser.add_argument('-v','--videos', nargs='+', type=int, required=True)
parser.add_argument('--model', help="restore checkpoint")
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

args = parser.parse_args()

print("videos: ", args.videos)

frame_stride = args.frame_stride
optical_flow_bound = 20.0

annotations_folder = "/home/ubuntu/data/mastoid_annotations"
video_folders = "/home/ubuntu/data/mastoid_video"
frame_folders = f"/home/ubuntu/data/mastoid_frames_{30//frame_stride}fps"
flow_folders = f"/home/ubuntu/data/mastoid_optical_flow_{30//frame_stride}fps"

print(frame_folders)
print(flow_folders)

videos = [f"V{i:03}" for i in args.videos]

width = 400
height = 225

padder = InputPadder((3, height, width))

flow_model = torch.nn.DataParallel(RAFT(args))
flow_model.load_state_dict(torch.load(args.model))

flow_model = flow_model.module
flow_model.to(DEVICE)
flow_model.eval()

def to_flow_model_input(img):
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    img = img[None].to(DEVICE)
    return padder.pad(img)[0]

def calc_flow(img1, img2):
    flow_low, flow_up = flow_model(img1, img2, iters=20, test_mode=True)
    flow = flow_up[0].cpu().numpy()
    flow = padder.unpad(flow)

    flow = np.round((flow + optical_flow_bound) / (2. * optical_flow_bound) * 255.)
    flow[flow < 0] = 0
    flow[flow > 255] = 255

    return flow


for video in videos:
    print(video)
    video_folder = os.path.join(video_folders, video)
    video_parts = glob.glob(os.path.join(video_folder, "*.[mM][pP]4"))
    video_parts.sort()
    
    annotation_file = os.path.join(annotations_folder, f"{video}.csv")
    annotation_df = pd.read_csv(annotation_file)

    # prev_end_idx = 0
    # for index, row in annotation_df.iterrows():
        # if index > 0 and prev_end_idx + 1 != row['Frame start']:
            # print(f"shoot video {video} {index} {prev_end_idx} {row['Frame start']}")
        # prev_end_idx = row['Frame end']

    row_idx = 0
    row = annotation_df.iloc[row_idx]

    # Frame index of the whole video.
    start_frame_idx = annotation_df.iloc[0]["Frame start"]
    end_frame_idx = annotation_df.iloc[-1]["Frame end"]
    print(f"start frame: {start_frame_idx}, end frame: {end_frame_idx}")

    # Open the first part.
    part_idx = 0
    cap = cv2.VideoCapture(video_parts[part_idx])
    print(video_parts[part_idx])
    part_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Frame index of the first video part.
    part_frame_idx = start_frame_idx
    cap.set(cv2.CAP_PROP_POS_FRAMES, part_frame_idx)
    
    # Frame metadata DF.
    frame_metadata_df = pd.DataFrame(columns=["Frame", "Step", "Task", "Img Path"])

    # Optical flow metadata DF.
    flow_metadata_df = pd.DataFrame(columns=["Frame", "Step", "Task", "Hori Path", "Vert Path"])

    frame_folder = os.path.join(frame_folders, video)
    if not os.path.exists(frame_folder):
        os.makedirs(frame_folder)

    flow_folder = os.path.join(flow_folders, video)
    if not os.path.exists(flow_folder):
        os.makedirs(flow_folder)

    # Get first frame.
    ret, prev_frame = cap.read()
    if not ret:
        print(f"Current Frame is {frame_idx} can't be read\n video {video_parts[part_idx]}\n part frame idx {part_frame_idx}")
    prev_frame = cv2.resize(prev_frame, (width, height))
    with torch.no_grad():
        prev_frame = to_flow_model_input(prev_frame)
    part_frame_idx += frame_stride
    
    for frame_idx in tqdm(range(start_frame_idx + frame_stride, end_frame_idx, frame_stride)):
        # Read frame.
        ret, frame = cap.read()
        if not ret:
            print(f"Current Frame is {frame_idx} can't be read\n video {video_parts[part_idx]}\n part frame idx {part_frame_idx}")
        else:
            # Save frame.
            frame_path = os.path.join(frame_folder, f"{frame_idx:06}.png")
            frame = cv2.resize(frame, (width, height))
            # cv2.imwrite(frame_path, frame)

            # Update frame metadata DF
            entry = pd.DataFrame.from_dict({
                 "Frame": [frame_idx],
                 "Step":  [row["Step"]],
                 "Task" : [row["Task"]],
                 "Img Path" : [frame_path]
            })
            
            frame_metadata_df = pd.concat([frame_metadata_df, entry], ignore_index=True)
            
            # Calculate optical flow and save flow.
            with torch.no_grad():
                frame = to_flow_model_input(frame)
                flow = calc_flow(prev_frame, frame)

            hori_path = os.path.join(flow_folder, f"{frame_idx:06}_hori.png")
            vert_path = os.path.join(flow_folder, f"{frame_idx:06}_vect.png")
            cv2.imwrite(hori_path, flow[0, ...])
            cv2.imwrite(vert_path, flow[1, ...])

            # Update optical flow metadata DF
            entry = pd.DataFrame.from_dict({
                 "Frame": [frame_idx],
                 "Step":  [row["Step"]],
                 "Task" : [row["Task"]],
                 "Hori Path" : [hori_path],
                 "Vert Path" : [vert_path]
            })
            flow_metadata_df = pd.concat([flow_metadata_df, entry], ignore_index=True)

            # Update previous frame.
            prev_frame = frame

        # Read next row if current one is finished.
        if frame_idx > row["Frame end"]:
            row_idx += 1
            row = annotation_df.iloc[row_idx]
        
        # Read next part if current one is finished.
        part_frame_idx += frame_stride
        if part_frame_idx >= part_frames:
            part_idx += 1
            if part_idx == len(video_parts):
                break
            # print(f"flow min_u = {optical_flow_min_u}, max_u = {optical_flow_max_u}, min_v = {optical_flow_min_v}, max_v = {optical_flow_max_v}")
            print(video_parts[part_idx])
            cap = cv2.VideoCapture(video_parts[part_idx])
            part_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            part_frame_idx = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, part_frame_idx)
    
    # print(f"flow min_u = {optical_flow_min_u}, max_u = {optical_flow_max_u}, min_v = {optical_flow_min_v}, max_v = {optical_flow_max_v}")
    
    frame_metadata_file = os.path.join(frame_folder, f"{video}.csv")
    frame_metadata_df.index.name = "Index"
    # frame_metadata_df.to_csv(frame_metadata_file)

    flow_metadata_file = os.path.join(flow_folder, f"{video}.csv")
    flow_metadata_df.index.name = "Index"
    flow_metadata_df.to_csv(flow_metadata_file)
