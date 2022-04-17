import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import pickle

"""
Unify all pickles into one
1. put all pickles in the pickles folder
2. change the netname below 
3. run, the unified pickle will be in root folder. Put it in the inputs folder
"""
netname = "TECNO"
# netname = "SVRC"
exports = []  # a list save all pk files from all videos
root = os.getcwd()
inputs_dir = os.path.join(root, "pickles")
outputs_dir = os.path.join(root, "outputs")
if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)

for filename in os.listdir(inputs_dir):
    if not filename.startswith("."):
        exports.append(pd.read_pickle(os.path.join(inputs_dir, filename)))
num_videos = len(exports)

pickleAll = open("{}.pkl".format(netname), "wb")
pickle.dump(exports, pickleAll)
pickleAll.close()
