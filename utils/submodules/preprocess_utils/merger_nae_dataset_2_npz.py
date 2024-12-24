from nae_core.utils.submodules.preprocess_utils.data_raw_reader import RoCatRLLabDataRawReader, RoCatNAEDataRawReader
# from utils.submodules.preprocess_utils.merger import RoCatRLLabDataMerger, RoCatNAEDataMerger
from nae_core.utils.submodules.training_utils.input_label_generator import InputLabelGenerator
from nae_core.utils.submodules.training_utils.data_loader import DataLoader as NAEDataLoader


import numpy as np
import matplotlib.pyplot as plt
import os
from python_utils.plotter import Plotter
from python_utils.printer import Printer
from collections import defaultdict


class NAEDatasetMerger:
    def __init__(self, ):
        self.util_printer = Printer()
        self.util_plotter = Plotter()
        # print('data files: ', data[0].files)    ['frame_num', 'time_step', 'position', 'quaternion']

    def load_data(self, data_dir):
        data = RoCatNAEDataRawReader(data_dir).read()
        data_raw = [traj['position'] for traj in data]
        return data_raw
    
    def plot_trajectory_dataset(self, data_raw, thrown_object, rotate_data_whose_y_up=False, num_data_to_plot=None):
        if num_data_to_plot is not None:
            # # get random 10 indices
            # indices = np.random.choice(len(data_raw), num_data_to_plot, replace=False)
            # trajectories = [data_raw[i] for i in indices]

            # no random
            trajectories = data_raw[:num_data_to_plot]
        else:
            trajectories = [traj for traj in data_raw]
        self.util_plotter.plot_trajectory_dataset(trajectories, title=thrown_object, rotate_data_whose_y_up=rotate_data_whose_y_up)

def main():
    data_bamboo_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/data/nae_paper_dataset/origin/trimmed_Paige_171'
    thrown_object = 'bamboo-ORIGIN'
    nae_data_merger = NAEDatasetMerger()
    data_raw = nae_data_merger.load_data(data_bamboo_dir)
    nae_data_merger.plot_trajectory_dataset(data_raw=data_raw, thrown_object=thrown_object, rotate_data_whose_y_up=False, num_data_to_plot=3)



if __name__ == '__main__':
    main()