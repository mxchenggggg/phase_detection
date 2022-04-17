import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import pickle
from pycm import ConfusionMatrix
import seaborn as sn
import numpy as np
"""
This is the file you run when you want to draw three bars in one plot. 
1. Put three single pkl file in the inputs folder. Must named as SVRC.pkl, TECNO.pkl, TRANS.pkl
Or you can hardcode file path in this script in main function. 
2. Run the script
3. Check results in outputs. (Will overwrite old ones)
"""


def rect(x, y, c, width=1, height=150):  # inputs: left lower (x,y) and color
    return matplotlib.patches.Rectangle((x, y), width, height, color=c)


def drawRibbonGraph(fig, ax, preds, labels, video_num, model_name):

    if model_name == "SVRC":
        pred_y = 200
        ax = fig.add_subplot(111)

    if model_name == "TeCNO":
        pred_y = 400

    if model_name == "Trans-SV":
        pred_y = 600

    label_y = 0
    #    fig = plt.figure()
    #    fig.set_size_inches(15.5, 7.5)

    for i, (pred, label) in enumerate(zip(preds, labels)):
        if pred == 0:
            rec_pred = rect(i, pred_y, "salmon")
        if pred == 1:
            rec_pred = rect(i, pred_y, "yellowgreen")
        if pred == 2:
            rec_pred = rect(i, pred_y, "cornflowerblue")
        # Only draw ground truth once, svrc is the first one
        if model_name == "SVRC":
            if label == 0:
                rec_label = rect(i, label_y, "salmon")
                ax.add_patch(rec_label)
            if label == 1:
                rec_label = rect(i, label_y, "yellowgreen")
                ax.add_patch(rec_label)
            if label == 2:
                rec_label = rect(i, label_y, "cornflowerblue")
                ax.add_patch(rec_label)
        ax.add_patch(rec_pred)
        # Only draw labels and legend onece
        if model_name == "Trans-SV":
            expose_label = mpatches.Patch(color="salmon", label="Expose")
            antrum_label = mpatches.Patch(color="yellowgreen", label="Antrum")
            fr_label = mpatches.Patch(
                color="cornflowerblue", label="Facial Recess")
            ax.legend(handles=[expose_label, antrum_label,
                      fr_label], fontsize=14)
            ax.set_xlabel("Num of frames", fontsize=16)
    return fig, ax


def drawRibbonGraphGT(fig, ax, preds, labels, video_num, model_name, label_y):
    for i, (pred, label) in enumerate(zip(preds, labels)):
        if label == 0:
            rec_label = rect(i, label_y, "salmon", height=150)
            ax.add_patch(rec_label)
        if label == 1:
            rec_label = rect(i, label_y, "yellowgreen", height=150)
            ax.add_patch(rec_label)
        if label == 2:
            rec_label = rect(i, label_y, "cornflowerblue", height=150)
            ax.add_patch(rec_label)
        expose_label = mpatches.Patch(color="salmon", label="Expose")
        antrum_label = mpatches.Patch(color="yellowgreen", label="Antrum")
        fr_label = mpatches.Patch(
            color="cornflowerblue", label="Facial Recess")
        ax.legend(handles=[expose_label, antrum_label,
                           fr_label], fontsize=14)
        ax.set_xlabel("Num of frames", fontsize=16)
    return fig, ax


class networkResults:
    def __init__(self, name, pickle):
        self.name = name
        (self.indxes, self.preds, self.labels) = self.extractPickle(pickle)

    def extractPickle(self, pickle):
        # return sorted preds and labels
        indexes = []
        preds = []
        labels = []
        for video in pickle:
            indexes.append(video[0])
            preds.append(video[1])
            labels.append(video[2])
        indexes_sorted = sorted(indexes)
        preds_sorted = [x for _, x in sorted(zip(indexes, preds))]
        labels_sorted = [x for _, x in sorted(zip(indexes, labels))]
        return indexes_sorted, preds_sorted, labels_sorted


def draw_all(svrc, tecno, trans):
    for k, video_num in enumerate(svrc.indxes):
        print("Drawing video {}....".format(video_num))
        fig = plt.figure()
        fig.set_size_inches(15.5, 10)
        fig, ax = drawRibbonGraph(
            fig, 0, svrc.preds[k], svrc.labels[k], video_num, svrc.name
        )
        fig, ax = drawRibbonGraph(
            fig, ax, tecno.preds[k], tecno.labels[k], video_num, tecno.name
        )
        fig, ax = drawRibbonGraph(
            fig, ax, trans.preds[k], trans.labels[k], video_num, trans.name
        )

        graphLength = len(trans.preds[k])
        y_labels = [75, 275, 475, 675]
        labels = ["GroundTruth", svrc.name, tecno.name, trans.name]
        plt.yticks(y_labels, labels, fontsize=16)
        plt.xlim([-100, graphLength + 500])
        plt.ylim([-100, 825])
        plt.title("Video {} segmentation result".format(
            video_num), fontsize=20)
        outname = "video_{}.png".format(video_num)
        fig.savefig(os.path.join(outputs_dir, outname))
        plt.close(fig)

        print("SVRC:")
        cm = ConfusionMatrix(
            actual_vector=svrc.preds[k], predict_vector=svrc.labels[k])
        cm.stat(overall_param=[], class_param=["ACC", "PPV", "TPR"])
        cm.print_normalized_matrix()
        target_class = ["Exposure", "Antrum", "Facial"]
        df_cm = pd.DataFrame(
            cm.to_array(normalized=True),
            index=target_class, columns=target_class)
        fig = plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True, cmap="YlGnBu")
        plt.xlabel("Predicted Label", fontdict={'fontsize': 22})
        plt.ylabel("True Label", fontdict={'fontsize': 22})
        plt.title(f"SVRC Video_{video_num} Confusion Matrix",
                  fontdict={'fontsize': 20})
        fig.savefig(os.path.join(
            outputs_dir, f"video_{video_num}_cm_svrc.png"))
        plt.close(fig)

        print("TECNO:")
        print("here", tecno.preds[k])
        cm = ConfusionMatrix(
            actual_vector=tecno.preds[k], predict_vector=tecno.labels[k])
        cm.stat(overall_param=[], class_param=["ACC", "PPV", "TPR"])
        cm.print_normalized_matrix()

        target_class = ["Exposure", "Antrum", "Facial"]
        df_cm = pd.DataFrame(
            cm.to_array(normalized=True),
            index=target_class, columns=target_class)
        fig = plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True, cmap="YlGnBu")
        plt.xlabel("Predicted Label", fontdict={'fontsize': 22})
        plt.ylabel("True Label", fontdict={'fontsize': 22})
        plt.title(f"TeCNO Video_{video_num} Confusion Matrix",
                  fontdict={'fontsize': 20})
        fig.savefig(os.path.join(
            outputs_dir, f"video_{video_num}_cm_tecno.png"))
        plt.close(fig)

        print("TRANS:")
        cm = ConfusionMatrix(
            actual_vector=trans.preds[k], predict_vector=trans.labels[k])
        cm.stat(overall_param=[], class_param=["ACC", "PPV", "TPR"])
        cm.print_normalized_matrix()

        target_class = ["Exposure", "Antrum", "Facial"]
        df_cm = pd.DataFrame(
            cm.to_array(normalized=True),
            index=target_class, columns=target_class)
        fig = plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True, cmap="YlGnBu")
        plt.xlabel("Predicted Label", fontdict={'fontsize': 22})
        plt.ylabel("True Label", fontdict={'fontsize': 22})
        plt.title(f"Trans-SVNet Video_{video_num} Confusion Matrix",
                  fontdict={'fontsize': 20})
        fig.savefig(os.path.join(
            outputs_dir, f"video_{video_num}_cm_trans.png"))
        plt.close(fig)

    print("Finish drawing total {} videos for all three models.".format(num_videos))


def draw_gt(svrc, tecno, trans):
    fig = plt.figure()
    fig.set_size_inches(15.5, 12)
    ax = fig.add_subplot(111)
    graphLength = 0
    for k, video_num in enumerate(svrc.indxes):
        print("Drawing video {}....".format(video_num))

        fig, ax = drawRibbonGraphGT(
            fig, ax, svrc.preds[k],
            svrc.labels[k],
            video_num, svrc.name, label_y=k * 200)

        graphLength = max(graphLength, len(svrc.preds[k]))
    labels = [f'V{i:03}' for i in [1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14]]
    y_labels = range(75, 75+len(labels) * 200, 200)
    plt.yticks(y_labels, labels, fontsize=16)
    plt.axis("equal")
    plt.ylim([-100, 100 + len(labels) * 200])
    plt.xlim([-100, graphLength + 100])
    plt.title("Ground Truth".format(
        video_num), fontsize=20)
    outname = "video_{}.png".format("all")
    fig.savefig(os.path.join(outputs_dir, outname))
    plt.close(fig)


if __name__ == "__main__":
    root = os.getcwd()
    inputs_dir = os.path.join(root, "inputs")
    outputs_dir = os.path.join(root, "outputs_tmp")
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    SVRC = pd.read_pickle(os.path.join(inputs_dir, "SVRC.pkl"))
    TECNO = pd.read_pickle(os.path.join(inputs_dir, "TECNO.pkl"))
    TRANS = pd.read_pickle(os.path.join(inputs_dir, "TransSpatialTest.pkl"))

    svrc = networkResults("SVRC", SVRC)
    tecno = networkResults("TeCNO", TECNO)
    trans = networkResults("Trans-SV", TRANS)
    net_list = [svrc, tecno, trans]

    for k, video_num in enumerate(svrc.indxes):
        print(np.array_equal(tecno.labels[k], trans.labels[k]))
        print(tecno.labels[k] - trans.labels[k])
    # assert svrc.indxes == tecno.indxes == trans.indxes  # check size
    # num_videos = len(svrc.indxes)
# 
    # plt.rcParams.update({'font.size': 22})
    # sn.set(font_scale=2)
# 
    # draw_gt(svrc, tecno, trans)
