#!/usr/bin/env python3

'''
Program description:
    - Read npz file and plot the trajectory of the object.
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import random
from python_utils.printer import Printer
from python_utils.plotter import Plotter

DEBUG = False
np.set_printoptions(suppress=True)

global_util_printer = Printer()

class RoCatDataRawReader:
    def __init__(self, file_path):
        self.trajectories = self.load_data(file_path)
        print('Loaded ', len(self.trajectories), ' trajectories')
    
    def load_data(self, data_folder):
        # load all npz data files
        # Find all npz files in the folder
        npz_files = [f for f in os.listdir(data_folder) if f.endswith('.npz')]
        print('Found ', len(npz_files), ' npz files in ', data_folder)
        data = []
        for f in npz_files:
            file_path = os.path.join(data_folder, f)
            traj = np.load(file_path, allow_pickle=True)
            # Convert numpy.lib.npyio.NpzFile to dict
            traj_dict = dict(traj)
            if 'preprocess' in traj_dict:
                # convert np.ndarray to dict
                traj_dict['preprocess'] = traj_dict['preprocess'].item()
            data.append(traj_dict)
        return data
    
    def read(self):
        return self.trajectories
    

    
# main
if __name__ == '__main__':
    # data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/data/nae_paper_dataset/data_preprocessed/Bamboo/data_train-134'
    data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/data/nae_paper_dataset/origin/trimmed_Bamboo_168'
    data_raw_reader = RoCatDataRawReader(data_dir)
    data = data_raw_reader.read()