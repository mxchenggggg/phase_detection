import pandas as pd
import os
import glob
import cv2
from tqdm import tqdm
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--frame_stride', type=int, required=True)
parser.add_argument('-v','--videos', nargs='+', type=int, required=True)

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

# videos = [f"V{i:03}" for i in range(1, 19)]
videos = [f"V{i:03}" for i in args.videos]

width = 400
height = 225

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
    prev_frame = cv2.resize(prev_frame, (int(width), int(height)), cv2.COLOR_RGB2BGR)
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    part_frame_idx += frame_stride

    # optical_flow_max_u = -float('inf')
    # optical_flow_max_v = -float('inf')

    # optical_flow_min_u = float('inf')
    # optical_flow_min_v = float('inf')

    for frame_idx in tqdm(range(start_frame_idx + frame_stride, end_frame_idx, frame_stride)):
        # Read frame.
        ret, frame = cap.read()
        if not ret:
            print(f"Current Frame is {frame_idx} can't be read\n video {video_parts[part_idx]}\n part frame idx {part_frame_idx}")
        else:
            # Save frame.
            frame_path = os.path.join(frame_folder, f"{frame_idx:06}.png")
            frame = cv2.resize(frame, (int(width), int(height)), cv2.COLOR_RGB2BGR)
            cv2.imwrite(frame_path, frame)

            # Update frame metadata DF
            entry = pd.DataFrame.from_dict({
                 "Frame": [frame_idx],
                 "Step":  [row["Step"]],
                 "Task" : [row["Task"]],
                 "Img Path" : [frame_path]
            })
            
            frame_metadata_df = pd.concat([frame_metadata_df, entry], ignore_index=True)
            
            # Calculate optical flow and save flow.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_frame, frame,
                                                None,
                                                0.5, 3, 15, 3, 5, 1.2, 0) 

            # optical_flow_min_u = min(optical_flow_min_u, np.min(flow[..., 0]))
            # optical_flow_min_v = min(optical_flow_min_v, np.min(flow[..., 1]))
            # optical_flow_max_u = max(optical_flow_max_u, np.max(flow[..., 0]))
            # optical_flow_max_v = max(optical_flow_max_v, np.max(flow[..., 1]))

            # if optical_flow_min_u < -optical_flow_bound or optical_flow_max_u > optical_flow_bound \
            #    or optical_flow_min_u < -optical_flow_bound or optical_flow_max_v > optical_flow_bound:
                # print(f"video part {part_idx} frame {part_frame_idx}, flow min_u = {optical_flow_min_u:.02f}, max_u = {optical_flow_max_u:.02f}, min_v = {optical_flow_min_v:.02f}, max_v = {optical_flow_max_v:.02f}")
                # print(f"avg_u: {np.mean(np.abs(flow[..., 0])):.02f}, avg_v: {np.mean(np.abs(flow[..., 1])):.02f}")
            
            flow = np.round((flow + optical_flow_bound) / (2. * optical_flow_bound) * 255.)
            flow[flow < 0] = 0
            flow[flow > 255] = 255

            hori_path = os.path.join(flow_folder, f"{frame_idx:06}_hori.png")
            vert_path = os.path.join(flow_folder, f"{frame_idx:06}_vect.png")
            cv2.imwrite(hori_path, flow[..., 0])
            cv2.imwrite(vert_path, flow[..., 1])

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
    frame_metadata_df.to_csv(frame_metadata_file)

    flow_metadata_file = os.path.join(flow_folder, f"{video}.csv")
    flow_metadata_df.index.name = "Index"
    flow_metadata_df.to_csv(flow_metadata_file)
