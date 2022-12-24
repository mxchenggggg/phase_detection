import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import pickle
from pycm import ConfusionMatrix
import seaborn as sn
import csv
import numpy as np

'''/**********************************************************************************************************
 * RUN THIS FILE TO DRAW ONE BAR and GT TOGETHER   AND CREATE CONFUSION MATRICES FOR EACH VIDEO IN MODEL. *
 *                                     ONE CSV FILES WILL BE GENERATED AS WELL.                           *
 *                                 1. PUT SINGLE PKL FILE IN THE INPUTS FOLDER.                           *
 *                  THIS PKL FILE SHOULD BE [(VIDEO_INDXES, PREDS, LABELS)...(.,.,.)...]                  *
 *                                    CHANGE THE PKL FILE NAME BELOW.                                     *
 *                                           2. RUN THE SCRIPT                                            *
 *     3. BAR PLOTS WILL BE IN OUTPUTS_BAR, CM PLOTS IN OUTPUTS_CM, MODEL SUMMARY IN OUTPUTS_SUMMARY      *
 **********************************************************************************************************/'''


'''/**********************************
 * CHANGE THE PKL FILE NAMES HERE *
 **********************************/ '''

model_pkl = "SVRC.pkl"
model_name = "SVRC_Net"


# matplot draw rectangle
def rect(x, y, c, width=1, height=75):  # inputs: left lower (x,y) and color
    return matplotlib.patches.Rectangle((x, y), width, height, color=c)


def drawRibbonGraph(fig, ax, preds, labels, video_num, model_name):
    pred_y = 95
    label_y = 0

    for i, (pred, label) in enumerate(zip(preds, labels)):
        if pred == 0:
            rec_pred = rect(i, pred_y, "salmon")
        if pred == 1:
            rec_pred = rect(i, pred_y, "yellowgreen")
        if pred == 2:
            rec_pred = rect(i, pred_y, "cornflowerblue")
        # Only draw ground truth once,model is the first one
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
        expose_label = mpatches.Patch(color="salmon", label="Expose")
        antrum_label = mpatches.Patch(color="yellowgreen", label="Antrum")
        fr_label = mpatches.Patch(
            color="cornflowerblue", label="Facial Recess")
        ax.legend(handles=[expose_label, antrum_label,
                           fr_label], fontsize=30)
        ax.set_xlabel("Num of frames", fontsize=16)
    return fig, ax


class networkResults:
    """ networkResults class
    Take in a pickle out put with format: [(VIDEO_INDXES, PREDS, LABELS)...(.,.,.)...] 
    Draw bars, CMs and write summary
    Args:
        name: network name
        pickle: pickle outputs of the model
        bar_dir: output dir for bar_graphs
        cm_dir: output dir for cm
        sum_dir: output dir for video summary
    """

    def __init__(self, name, pickle, bar_dir, cm_dir, sum_dir):
        self.name = name
        self.bar_dir = bar_dir
        self.cm_dir = cm_dir
        self.sum_dir = sum_dir
        (self.indxes, self.preds, self.labels) = self.extractPickle(pickle)

    #  Sort pickle and return three sorted list indexes = [1,2,3...] preds = [pred1,pred2..] labels = [label1..]
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

    # args, video num, k is the index in the list
    def drawCM(self, video_num, k):
        cm = ConfusionMatrix(
            actual_vector=self.labels[k],
            predict_vector=self.preds[k])
        A = cm.ACC
        P = cm.PPV
        R = cm.TPR
        summary_line = [
            f'{video_num}', f'{cm.Overall_ACC}', f'{A.get(0)}', f'{A.get(1)}',
            f'{A.get(2)}', f'{P.get(0)}', f'{P.get(1)}', f'{P.get(2)}',
            f'{R.get(0)}', f'{R.get(1)}', f'{R.get(2)}']
        with open(f"{self.sum_dir}/{self.name}_summary.csv", 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(summary_line)

        # Print to console
        #cm.stat(overall_param=[], class_param=["ACC", "PPV", "TPR"])

        target_class = ["Exposure", "Antrum", "Facial"]
        df_cm = pd.DataFrame(
            cm.to_array(normalized=True),
            index=target_class, columns=target_class)
        fig = plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True, cmap="YlGnBu")
        plt.xlabel("Predicted Label", fontdict={'fontsize': 22})
        plt.ylabel("True Label", fontdict={'fontsize': 22})
        plt.title(f"{model_name} Video_{video_num} Confusion Matrix",
                  fontdict={'fontsize': 20})
        fig.savefig(os.path.join(
            self.cm_dir, f"{self.name}_video_{video_num}_cm.png"))
        plt.close(fig)


if __name__ == "__main__":
    root = os.getcwd()
    inputs_dir = os.path.join(root, "inputs")
    bar_dir = os.path.join(root, "outputs_bar")
    cm_dir = os.path.join(root, "outputs_cm")
    sum_dir = os.path.join(root, "outputs_summary")

    if not os.path.exists(bar_dir):
        os.makedirs(bar_dir)
    if not os.path.exists(cm_dir):
        os.makedirs(cm_dir)
    if not os.path.exists(sum_dir):
        os.makedirs(sum_dir)

    MODEL = pd.read_pickle(os.path.join(inputs_dir, model_pkl))

    model = networkResults(model_name, MODEL, bar_dir, cm_dir, sum_dir)

    net_list = [model]  # single net
    num_videos = len(model.indxes)
    # Write the output summary header
    header = ['Video Number', 'Overall Acc', 'Acc_expose', 'Acc_antrum',
              'Acc_facial', 'Prc_expose', 'Prc_antrum', 'Prc_facial',
              'Rc_expose', 'Rc_antrum', 'Rc_facial']
    for net in net_list:
        with open(f"{sum_dir}/{net.name}_summary.csv", 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    plt.rcParams.update({'font.size': 22})
    sn.set(font_scale=2)

    for k, video_num in enumerate(model.indxes):
        print("Drawing video {}....".format(video_num))
        tmp_pred = []
        tmp_label = []
        for t in model.preds[k]:
            tmp_pred += [t]*3
        for t in model.labels[k]:
            tmp_label += [t]*3

        fig = plt.figure()
        fig.set_size_inches(35, 10)
        ax = fig.add_subplot(111)
        fig, ax = drawRibbonGraph(
            fig, ax, np.array(tmp_pred),
            np.array(tmp_label),
            video_num, model.name)

        graphLength = len(tmp_label)
        y_labels = [38, 133]
        labels = ["GroundTruth", model.name]
        plt.yticks(y_labels, labels, fontsize=30)
        plt.xlim([-100, graphLength + 500])
        plt.ylim([-100, 400])
        plt.title("Video {} segmentation result".format(
            video_num), fontsize=30)
        outname = "video_{}.png".format(video_num)
        fig.savefig(os.path.join(bar_dir, outname))
        plt.close(fig)

        # model.drawCM(video_num, k)

    print("Finish drawing and construct CM for total {} videos for all three models.".format(num_videos))
