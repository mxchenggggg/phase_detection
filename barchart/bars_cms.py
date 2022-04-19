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

'''/**********************************************************************************************************
 * RUN THIS FILE TO DRAW THREE BARS IN ONE CHART AND CREATE CONFUSION MATRICES FOR EACH VIDEO EACH MODEL. *
 *                           THREE SUMMARY CSV FILES WILL BE GENERATED AS WELL.                           *
 *                           1. PUT THREE SINGLE PKL FILE IN THE INPUTS FOLDER.                           *
 *                  EACH PKL FILE SHOULD BE [(VIDEO_INDXES, PREDS, LABELS)...(.,.,.)...]                  *
 *                                    CHANGE THE PKL FILE NAME BELOW.                                     *
 *                                           2. RUN THE SCRIPT                                            *
 *     3. BAR PLOTS WILL BE IN OUTPUTS_BAR, CM PLOTS IN OUTPUTS_CM, MODEL SUMMARY IN OUTPUTS_SUMMARY      *
 **********************************************************************************************************/'''


'''/**********************************
 * CHANGE THE PKL FILE NAMES HERE *
 **********************************/ '''

#svrc pkl filename 
svrc_pkl = "SVRC.pkl"
#tecno pkl filename
tecno_pkl = "TECNO.pkl"
#transSV pkl filename
trans_pkl = "TRANS_seqlen=10.pkl"

# matplot draw rectangle
def rect(x, y, c, width=1, height=150):  # inputs: left lower (x,y) and color
    return matplotlib.patches.Rectangle((x, y), width, height, color=c)


def drawRibbonGraph(fig, ax, preds, labels, video_num, model_name):

    if model_name == "SVRC":
        pred_y = 200

    if model_name == "TeCNO":
        pred_y = 400

    if model_name == "Trans-SV":
        pred_y = 600

    label_y = 0

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
    def __init__(self, name, pickle,bar_dir,cm_dir,sum_dir):
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
    
    #args, video num, k is the index in the list
    def drawCM(self,video_num,k):
        cm = ConfusionMatrix(actual_vector=self.labels[k], predict_vector=self.preds[k])
        A = cm.ACC
        P = cm.PPV
        R = cm.TPR
        summary_line = [f'{video_num}', f'{cm.Overall_ACC}', f'{A.get(0)}', f'{A.get(1)}', f'{A.get(2)}',f'{P.get(0)}',
                        f'{P.get(1)}',f'{P.get(2)}',f'{R.get(0)}',f'{R.get(1)}',f'{R.get(2)}']
        with open(f"{self.sum_dir}/{self.name}_summary.csv", 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(summary_line)
            
        #Print to console
        #cm.stat(overall_param=[], class_param=["ACC", "PPV", "TPR"])

        target_class = ["Exposure", "Antrum", "Facial"]
        df_cm = pd.DataFrame(
            cm.to_array(normalized=True), index=target_class, columns=target_class)
        fig = plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True, cmap="YlGnBu")
        plt.xlabel("Predicted Label", fontdict={'fontsize': 22})
        plt.ylabel("True Label", fontdict={'fontsize': 22})
        plt.title(f"SVRC Video_{video_num} Confusion Matrix",
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

    SVRC = pd.read_pickle(os.path.join(inputs_dir, svrc_pkl))
    TECNO = pd.read_pickle(os.path.join(inputs_dir, tecno_pkl))
    TRANS = pd.read_pickle(os.path.join(inputs_dir, trans_pkl))

    svrc = networkResults("SVRC", SVRC,bar_dir,cm_dir,sum_dir)
    tecno = networkResults("TeCNO", TECNO,bar_dir,cm_dir,sum_dir)
    trans = networkResults("Trans-SV", TRANS,bar_dir,cm_dir,sum_dir)
    net_list = [svrc, tecno, trans]
    assert svrc.indxes == tecno.indxes == trans.indxes  # check size
    num_videos = len(svrc.indxes)
    # Write the output summary header
    header = ['Video Number', 'Overall Acc', 'Acc_expose', 'Acc_antrum', 'Acc_facial','Prc_expose','Prc_antrum','Prc_facial','Rc_expose','Rc_antrum','Rc_facial']
    for net in net_list:
        with open(f"{sum_dir}/{net.name}_summary.csv", 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    plt.rcParams.update({'font.size': 22})
    sn.set(font_scale=2)

    for k, video_num in enumerate(svrc.indxes):
        print("Drawing video {}....".format(video_num))
        fig = plt.figure()
        fig.set_size_inches(15.5, 10)
        ax = fig.add_subplot(111)
        fig, ax = drawRibbonGraph(
            fig, ax, svrc.preds[k], svrc.labels[k], video_num, svrc.name
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
        fig.savefig(os.path.join(bar_dir, outname))
        plt.close(fig)

        svrc.drawCM(video_num,k)
        tecno.drawCM(video_num,k)
        trans.drawCM(video_num,k)

    print("Finish drawing and construct CM for total {} videos for all three models.".format(num_videos))

