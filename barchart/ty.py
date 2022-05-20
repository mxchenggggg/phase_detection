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
import torch 


model_pkl = "prediction_results.pkl"
model_name = "SVRC_Net"
root = os.getcwd()
inputs_dir = os.path.join(root, "inputs")
pickle = pd.read_pickle(os.path.join(inputs_dir, model_pkl))
print(pickle[1])