import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import sys

"""
1. Put pickle files (for all videos you want to process) in inputs folder, 
the first item in each pickle file should be the prediction 
the second should be the ground truth
2. Put your video numbers in the video list
3. check output folders
"""


def rect(x, y, c, width=1, height=150):  # inputs: left lower (x,y) and color
    return matplotlib.patches.Rectangle((x, y), width, height, color=c)


def drawRibbonGraph(preds, labels, video_num, out_dir):
    pred_y = 0
    label_y = 200
    fig = plt.figure()
    fig.set_size_inches(15.5, 7.5)
    ax = fig.add_subplot(111)
    for i, (pred, label) in enumerate(zip(preds, labels)):
        if pred == 0:
            rec_pred = rect(i, pred_y, "salmon")
        if pred == 1:
            rec_pred = rect(i, pred_y, "yellowgreen")
        if pred == 2:
            rec_pred = rect(i, pred_y, "cornflowerblue")

        if label == 0:
            rec_label = rect(i, label_y, "salmon")
        if label == 1:
            rec_label = rect(i, label_y, "yellowgreen")
        if label == 2:
            rec_label = rect(i, label_y, "cornflowerblue")
        ax.add_patch(rec_pred)
        ax.add_patch(rec_label)
    expose_label = mpatches.Patch(color="salmon", label="Expose")
    antrum_label = mpatches.Patch(color="yellowgreen", label="Antrum")
    fr_label = mpatches.Patch(color="cornflowerblue", label="Facial Recess")
    ax.legend(handles=[expose_label, antrum_label, fr_label])
    y_labels = [75, 275]
    labels = ["Prediction", "GroundTruth"]
    labels = ["Prediction", "GroundTruth"]
    plt.yticks(y_labels, labels)
    plt.xlim([-100, 2500])
    plt.ylim([-100, 700])
    ax.set_xlabel("Num of frames")
    outname = "video_{}.png".format(video_num)
    fig.savefig(os.path.join(out_dir, outname))


if __name__ == "__main__":
    args = sys.argv
    video_list = []
    if len(args) == 1:
        raise Exception("Give the video num you are processing! like 1 2 3 4..")
    for i in range(1, len(args)):
        video_list.append(args[i])
    print("The video we are processing are: {}".format(video_list))

    exports = []  # a list save all pk files from all videos
    root = os.getcwd()
    inputs_dir = os.path.join(root, "inputs")
    outputs_dir = os.path.join(root, "outputs")
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    for filename in os.listdir(inputs_dir):
        if not filename.startswith("."):
            exports.append(pd.read_pickle(os.path.join(inputs_dir, filename)))
    num_videos = len(exports)
    assert len(video_list) == num_videos  # inputs and vido_list size check
    for k, video_num in enumerate(video_list):
        preds = exports[k][0]  # prediction is the first in each pickle
        labels = exports[k][1]  # label is the second in each pickle
        assert len(preds) == len(labels)  # label and preds same size
        print("Drawing video {}....".format(video_num))
        drawRibbonGraph(preds, labels, video_num, outputs_dir)
    print("Finish drawing total {} videos.".format(num_videos))
