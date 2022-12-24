import pandas as pd
import os

data_root = "/home/ubuntu/data/mastoid_400225_downsampled/"
all_annotations_columns = ["path", "class", "vid_idx"]

usable_tool_labels = [0, 1, 2, 3, 4]
unusable_tool_labels = [5]


def vid_name(vid_idx):
    return f"V{vid_idx:03}"


def annot_file(vid_idx):
    return f"{vid_name(vid_idx)}_tool_annotation.csv"


def annot_abs_path(vid_idx):
    vid_folder = os.path.join(data_root, vid_name(vid_idx))
    return os.path.join(vid_folder, annot_file(vid_idx))


def all_annot_abs_path():
    return os.path.join(data_root, "tool_annotations.csv")


def merge_all_annotations(vid_idxes):
    all_annotations = pd.DataFrame(columns=all_annotations_columns)
    for vid_idx in vid_idxes:
        file_path = annot_abs_path(vid_idx)
        df = pd.read_csv(file_path)
        df = df[["path", "class"]]
        df = df[df["class"].isin(usable_tool_labels + unusable_tool_labels)]
        df["vid_idx"] = vid_idx
        all_annotations = all_annotations.append(df)

    print(f"{len(all_annotations)} frames for video:")
    print(vid_idxes)
    all_annot_file_path = all_annot_abs_path()
    all_annotations.to_csv(all_annot_file_path)
    print(f"write all annotations to {all_annot_file_path}")


def main():
    merge_all_annotations([1, 2, 3, 4, 13, 14])


if __name__ == "__main__":
    main()
