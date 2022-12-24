import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import Normalize
import pandas as pd
import pickle
import os

# pred_dir = "/home/wluo14/data/prediction/WILDCAT/220624-090613_testing"
pred_dir = "/home/wluo14/data/prediction/WILDCAT/220629-202701_testing_0629"

pred_metadata_file = os.path.join(pred_dir, "WILDCAT_WSL_maps_metadata.csv")
maps_normalize_file = os.path.join(pred_dir, "WILDCAT_WSL_maps_normalize.txt")
labels = ["cut_on", "cut_off", "diamond_on", "diamond_off", "no_drill"]
axes_names = labels + ["image", "pred", "target"]
layout = [['image', 'image', 'image', 'image', 'target', 'target'],
          ['image', 'image', 'image', 'image', 'target', 'target'],
          ['image', 'image', 'image', 'image', 'pred', 'pred'],
          ['image', 'image', 'image', 'image', 'pred', 'pred'],
          ['cut_on', 'cut_off', 'diamond_on', 'diamond_off', 'no_drill', '.'],
          ]
title_font = 8


class Visualizer:
    def __init__(self) -> None:
        self.ind = 0

        self.df = pd.read_csv(pred_metadata_file)
        
        with open(maps_normalize_file, 'r') as f:
            vmin, vmax = map(float, f.readline().split())
        norm = Normalize(vmin=vmin, vmax=vmax)
    

        self.fig, self.axarr = plt.subplot_mosaic(layout)
        for _, ax in self.axarr.items():
            ax.set_axis_off()

        self.fig.canvas.mpl_connect('key_press_event', self.on_press)

        self.images_raw, pred_label, target_label = self.get_data(self.ind)
        titles = labels + ["Original Image", f"Predicted label: {pred_label}",
                           f"Target (GT) label: {target_label}"]
        self.images = {}
        for i, name in enumerate(axes_names):
            self.images[name] = self.axarr[name].imshow(
                self.images_raw[name], norm=norm) #cmap='gray'
            self.axarr[name].set_title(titles[i], fontsize=title_font)

        self.fig.colorbar(self.images["pred"], ax=self.axarr["pred"])
        self.fig.colorbar(self.images["target"], ax=self.axarr["target"])

    def get_data(self, i):
        path = self.df.loc[i, "img_path"]
        path = path.replace("ubuntu", "wluo14")
        print(path)
        image = mpimg.imread(path)

        path = self.df.loc[i, "map_path"]
        path = path.replace("ubuntu", "wluo14")
        with open(path, 'rb') as file:
            map_data = pickle.load(file)

        pred_idx = int(np.argmax(map_data["pred"]))
        target_idx = int(map_data["target"])

        maps = map_data['maps']
        images_raw = {
            "image": image,
            "pred": maps[pred_idx],
            "target": maps[target_idx]}

        for i, label in enumerate(labels):
            images_raw[label] = maps[i]

        return images_raw, labels[pred_idx], labels[target_idx]

    def len(self):
        return len(self.df)

    def update(self, i):
        print(i)
        self.images_raw, pred_label, target_label = self.get_data(i)
        for name in axes_names:
            self.images[name].set_data(self.images_raw[name])

        self.axarr["pred"].set_title(
            f"Predicted label: {pred_label}", fontsize=title_font)
        self.axarr["target"].set_title(
            f"Target (GT) label: {target_label}", fontsize=title_font)

        plt.draw()

    def next(self, event):
        self.ind += 1
        i = self.ind % self.len()
        self.update(i)

    def prev(self, event):
        self.ind -= 1
        i = self.ind % self.len()
        self.update(i)

    def on_press(self, event):
        if event.key == 'left':
            self.prev(event)
            print("previous")
        elif event.key == 'right':
            self.next(event)
            print("next")

def main():
    visualizer = Visualizer()
    plt.show()

if __name__ == "__main__":
    main()