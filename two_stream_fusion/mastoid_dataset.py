from two_stream_model import TwoStreamFusion
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import os
import cv2
import numpy as np
import pandas as pd
from copy import deepcopy

import torchvision.transforms.functional as TF
from torchvision import transforms

mastoid_frames = "/home/ubuntu/data/mastoid_frames"
mastoid_flow = "/home/ubuntu/data/mastoid_optical_flow_5fps"

# phases = {"Expose": 0, "Antrum": 1, "Facial_recess": 2, "Idle": 3}
# actions = {"Tegmen" : 0, "SS" : 1, "EAC" : 2, "Open_antrum" : 3, "Facial_recess" : 4, "Idle" : 5, "Microscope_zoom" : 6, "Microscope_move" : 6}
actions = {"Tegmen": 0, "SS": 1, "EAC": 2, "Open_antrum": 3, "Facial_recess": 4}

phases = {"Expose": 0, "Antrum": 1, "Facial_recess": 2}
# actions = {"Tegmen": 0, "SS": 1, "EAC": 2, "Open_antrum": 3, "Expose_incus": 4, "Facial_recess": 5}

phase_actions = {"Expose": ["Tegmen", "SS", "EAC"], "Antrum" : ["Open_antrum", "Expose_incus"], "Facial_recess" : ["Facial_recess"]}

num_class = {"Step": len(phases), "Task": len(actions)}
labels = {"Step": phases, "Task": actions}

class MastoidTransform:
    def __init__(
            self, augment, hflip_p=None, affine_p=None, rotate_angle=None,
            scale_range=None, color_jitter_p=None, brightness=0.0, contrast=0.0,
            saturation=0.0, hue=0.0, flow_clip=20.0) -> None:
        # Peform data augmentation is set to True.
        self.augment = augment

        # Horizontal flip
        self.hflip_p = hflip_p

        # Rotate + Scale
        self.affine_p = affine_p
        self.rotate_angle = rotate_angle
        self.scale_range = scale_range

        # Color jittering
        self.color_jitter_p = color_jitter_p
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue)

        # RGB normalization
        self.rgb_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        self.flow_clip = flow_clip

    def __call__(self, rgb_frames: torch.Tensor, flow_stacks: torch.Tensor):
        # Input shapes:
        #   rgb_frames  (T * 3 * W * H)
        #   flow_stacks (T * 20 * W * H)

        rgb_frames = rgb_frames / 255.

        if self.augment:
            # Horizontal flip
            hflip = np.random.random()
            if self.hflip_p is not None and hflip < self.hflip_p:
                rgb_frames = TF.hflip(rgb_frames)
                flow_stacks = TF.hflip(flow_stacks)

            # Rotate + scale
            affine = np.random.random()
            if self.affine_p is not None and affine < self.affine_p:
                angle = np.random.uniform(-self.rotate_angle, self.rotate_angle)
                scale = np.random.uniform(
                    self.scale_range[0],
                    self.scale_range[1])

                rgb_frames = TF.affine(
                    rgb_frames, angle=angle, translate=(0.0, 0.0),
                    shear=0.0, scale=scale)
                flow_stacks = TF.affine(
                    flow_stacks, angle=angle, translate=(0.0, 0.0),
                    shear=0.0, scale=scale)

            # Color jittering, only for RGB images
            color_jitter = np.random.random()
            if self.color_jitter_p is not None and color_jitter < self.color_jitter_p:
                rgb_frames = self.color_jitter(rgb_frames)

        # Normalize RGB frames
        rgb_frames = self.rgb_normalize(rgb_frames)

        # Rescale back flows to [-flow_clip, flow_clip]
        flow_stacks = flow_stacks / 255. * (2 * self.flow_clip) - self.flow_clip

        return rgb_frames, flow_stacks


class MastoidTwoSteamDataset(Dataset):
    def __init__(
        self, name, transform, fps, videos, rgb_frames, opf_frames,
            group_mode, class_mode, flow_clip=20.0):
        print(f'{name} dataset')

        self.videos = [f"V{i:03}" for i in videos]
        self.group_mode = group_mode  # video, class

        self.class_mode = class_mode # Task, Step

        self.flow_clip = flow_clip

        self.rgb_frames = rgb_frames
        self.opf_frames = opf_frames
        self.half_opf_frames = opf_frames // 2
        self.sample_stride = opf_frames

        self.labels = labels[self.class_mode]
        self.num_class = num_class[self.class_mode]

        self.transformation = transform

        self.root = {
            "rgb": f"/home/ubuntu/data/mastoid_frames_{fps}fps",
            "flow": f"/home/ubuntu/data/mastoid_optical_flow_{fps}fps"}

        self.dfs = {
            "rgb": pd.DataFrame(
                columns=["Index", "Frame", "Step", "Task", "Img Path"]),
            "flow": pd.DataFrame(
                columns=["Index", "Frame", "Step", "Task", "Hori Path", "Vert Path"])
        }

        # Each sample consists of 5 rgb frames 5 frames apart and 5 optical
        # flow stack consisting of 10 frames center at each rgb frame, spanning
        # 50 frames in total.

        # index refers to index of the metadata pandas Dataframe.

        # {video : {lable_1 : df_idxes, ..., lable_n : df_idxes}}
        videos_class_idxes = {}

        class_count = {label: 0 for label, _ in self.labels.items()}

        for video in self.videos:
            cur_df_len = len(self.dfs["rgb"])
            for stream in ["rgb", "flow"]:
                df = pd.read_csv(os.path.join(
                    self.root[stream],
                    video, f"{video}.csv"))
                self.dfs[stream] = pd.concat(
                    [self.dfs[stream], df], ignore_index=True)

            # {lable_1 : df_idxes, ..., lable_n : df_idxes}
            video_class_idxes = {label: np.empty(0, dtype=int)
                                 for label, _ in self.labels.items()}
            # Find adjacent rows of same label.
            adj_check = (df[self.class_mode] != df[self.class_mode].shift()).cumsum()
            groups_by_label = df[['Index', 'Step', 'Task']].groupby(adj_check).agg(list)

            for _, row in groups_by_label.iterrows():
                label = row[self.class_mode][0]
                if label not in self.labels:
                    continue

                # if self.class_mode == "Step" and row['Task'][0] not in phase_actions[label]:
                    # continue

                idxes = np.array(row['Index'][self.half_opf_frames: -(self.opf_frames * (
                    self.rgb_frames + 1) + self.half_opf_frames - 1)], dtype=int)
                
                if len(idxes) == 0:
                    continue

                idx_idxes = range(0, len(idxes), self.sample_stride)
                # idx_noise = np.random.randint(-self.sample_stride // 2, self.sample_stride // 2 + 1, len(idx_idxes))
                # idx_noise[0] = 0
                # idx_noise[-1] = 0
                # idx_idxes += idx_noise
                
                idxes = idxes[idx_idxes] + cur_df_len

                video_class_idxes[label] = np.append(
                    video_class_idxes[label], idxes)
                class_count[label] += len(idxes)

            videos_class_idxes[video] = video_class_idxes

        # Used for debug.
        self.df_length = len(self.dfs["rgb"])

        # Find minimum number of samples of a class.
        min_class_count = len(self.dfs["rgb"]) + 1
        original_class_count_str = "Original dataset: "
        for label, count in class_count.items():
            original_class_count_str += f"({label}, {count}) "
            min_class_count = min(min_class_count, count)
        print(original_class_count_str)

        balanced_class_count = {label: 0 for label, _ in self.labels.items()}
        self.dataset_length = 0

        # For each video and each label, select some samples such that
        # labels are balanced across whole dataset.
        for video, video_class_idxes in videos_class_idxes.items():
            for label, idxes in video_class_idxes.items():
                portion = min_class_count / class_count[label]
                num_samples = int(portion * len(idxes))

                sampled_idxes = idxes[np.random.permutation(
                    len(idxes))[:num_samples]]
                videos_class_idxes[video][label] = sampled_idxes

                balanced_class_count[label] += num_samples
                self.dataset_length += num_samples

        class_count = balanced_class_count
        balanced_class_count_str = "Balanced dataset: "
        for label, count in class_count.items():
            balanced_class_count_str += f"({label}, {count}) "
        print(balanced_class_count_str)
        print(f'Dataset length: ', self.dataset_length)

        if self.group_mode == "video":
            # {video_1 : df_idxes, ..., video_n : df_idxes}
            video_idxes = {video: np.empty(0, dtype=int)
                           for video in self.videos}
            for video, video_class_idxes in videos_class_idxes.items():
                for label, idxes in video_class_idxes.items():
                    video_idxes[video] = np.append(video_idxes[video], idxes)

            video_samples_str = "Number of samples per video: "
            for video, idxes in video_idxes.items():
                video_samples_str += f"({video}, {len(idxes)}) "
            print(video_samples_str)

            self.group_idxes = video_idxes
        else:  # self.group_mode == "class"
            # {lable_1 : df_idxes, ..., lable_n : df_idxes}
            class_idxes = {label: np.empty(0, dtype=int)
                           for label, _ in self.labels.items()}

            for video, video_class_idxes in videos_class_idxes.items():
                for label, idxes in video_class_idxes.items():
                    class_idxes[label] = np.append(class_idxes[label], idxes)

            self.group_idxes = class_idxes

    def __getitem__(self, index):
        # assert index >= 0 and index < self.df_length, f'Invalid df idx {index}, maximum idx: {self.df_length}'
        label_name = self.dfs["rgb"].iloc[index][self.class_mode]
        # assert label_name in self.labels, f'Invalid label {label_name}'

        label_idx = self.labels[label_name]
        label_one_hot = np.zeros(num_class[self.class_mode], dtype=np.float32)
        label_one_hot[label_idx] = 1.

        # stride = np.random.randint(1, self.opf_frames)
        stride = self.opf_frames
        rgb_frames = self.read_rgb(index, stride)
        flow_stacks = self.read_flow(index, stride)

        rgb_frames, flow_stacks = self.transformation(rgb_frames, flow_stacks)

        return rgb_frames, flow_stacks, torch.from_numpy(label_one_hot)
        # return torch.from_numpy(label_one_hot)

    def __len__(self):
        return self.dataset_length

    def read_rgb(self, df_idx, stride):
        frames = []
        for idx in range(df_idx, df_idx + self.rgb_frames * stride, stride):
            # assert idx >= 0 and idx < self.df_length, f'Invalid df idx {idx}, maximum idx: {self.df_length}'
            rgb = cv2.cvtColor(
                cv2.imread(self.dfs["rgb"].iloc[idx]['Img Path']),
                cv2.COLOR_BGR2RGB)
            frames.append(rgb)
        # T * C * W * H
        frames = np.asarray(frames, dtype=np.float32).transpose([0, 3, 1, 2])
        return torch.from_numpy(frames)

    def read_flow(self, df_idx,stride):
        flow_stacks = []
        for rgb_idx in range(
                df_idx, df_idx + self.rgb_frames * stride, stride):
            flow_stack = []
            for idx in range(
                    rgb_idx - self.half_opf_frames, rgb_idx + self.half_opf_frames):
                # assert idx >= 0 and idx < self.df_length, f'Invalid df idx {idx}, maximum idx: {self.df_length}'

                flow_u = cv2.imread(
                    self.dfs["flow"].iloc[idx]['Hori Path'],
                    cv2.IMREAD_GRAYSCALE).astype(
                    np.float32)
                flow_stack.append(flow_u)
                flow_v = cv2.imread(
                    self.dfs["flow"].iloc[idx]['Vert Path'],
                    cv2.IMREAD_GRAYSCALE).astype(
                    np.float32)
                flow_stack.append(flow_v)
            flow_stack = np.stack(flow_stack)
            flow_stacks.append(flow_stack)
        # T * C * W * H, C = 2 * ops_frames
        flow_stacks = np.asarray(flow_stacks, dtype=np.float32)
        return torch.from_numpy(flow_stacks)


class BatchSampler():
    def __init__(self, group_idxes, batch_size, shuffle=True, debug=False):
        self.max_group_samples = np.max([len(idxes)
                                         for _, idxes in group_idxes.items()])
        self.num_groups = len(group_idxes)
        self.idxes_with_invalid = np.ones(
            (self.num_groups, self.max_group_samples), dtype=int) * -1
        self.samples_count = 0

        for i, (_, idxes) in enumerate(group_idxes.items()):
            self.idxes_with_invalid[i, :len(idxes)] = idxes
            self.samples_count += len(idxes)

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_batches = int(np.ceil(self.samples_count / self.batch_size))
        print(
            f'Number of samples {self.samples_count}, number of batches {self.num_batches}')

        # Only for debugging.
        self.debug = debug
        if self.debug:
            self.debug_idxes = []
            for _, idxes in group_idxes.items():
                self.debug_idxes += idxes.tolist()
            self.debug_idxes.sort()
            self.debug_idxes = np.array(self.debug_idxes)

    def __iter__(self):
        idxes = deepcopy(self.idxes_with_invalid)
        if self.shuffle:
            # Shuffle videos.
            group_idxes = np.random.permutation(self.num_groups)
            idxes = self.idxes_with_invalid[group_idxes, :]

            # Shuffle samples for each video.
            for i in range(self.num_groups):
                samples_idx = np.random.permutation(self.max_group_samples)
                idxes[i] = idxes[i, samples_idx]

        # Flatten and ignore invalid indexes.
        idxes = idxes.transpose().flatten()
        idxes = idxes[idxes >= 0]

        # Only for debugging.
        if self.debug:
            sorted_idxes = np.sort(idxes)
            if (sorted_idxes != self.debug_idxes).any():
                print("Batch sampler invalid indexes!")
                for i in range(len(idxes)):
                    print(sorted_idxes[i], self.debug_idxes[i])

        batch_idxes = [batch.tolist()
                       for batch in np.array_split(idxes, self.num_batches)]
        return iter(batch_idxes)

    def __len__(self):
        return self.num_batches


if __name__ == "__main__":

    train_transform = MastoidTransform(
        augment=True, hflip_p=0.5, affine_p=0.5, rotate_angle=0.,
        scale_range=(0.9, 1.1),
        color_jitter_p=0.5, brightness=0.2, contrast=0.2,
        saturation=0.2, hue=0.1)

    videos = [1, 5, 6, 7 ,8 ,10, 11, 12]
    dataset = MastoidTwoSteamDataset(
        "train", train_transform, 15, videos, 5, 10, "class", "Step")
    dataloader = DataLoader(dataset,
                            batch_sampler=BatchSampler(
                                dataset.group_idxes, 32, debug=True),
                            num_workers=32)

    # model = TwoStreamFusion(5, 10, 225, 400)
    # model.cuda()
    # model = nn.DataParallel(model)

    # for x in tqdm(dataloader):
        # rgb, flow, label_one_hot = x

        # print(rgb.shape, flow.shape, label_one_hot.shape)
        # print(rgb[0,0,0,100,100:110])
        # print(flow[0,0,0,100,100:110])
        # print(label_one_hot[0])

        # rgb = Variable(rgb.cuda())
        # flow = Variable(flow.cuda())

        # out = model((rgb, flow))

    # for vs in [[1, 5], [2, 12], [4, 11], [8, 10], [6, 7]]:
        # print(f'video {vs}')
        # dataset = MastoidTwoSteamDataset(
        # "train", train_transform, 15, vs, 5, 10, "class", "Step")
