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

    def read_position(self, ):
        pos_data = [traj['position'] for traj in self.trajectories]
        return pos_data
    
class RoCatRLLabDataRawReader:
    def __init__(self, file_path):
        # load the npz file
        self.trajectories = np.load(file_path, allow_pickle=True)['trajectories']
        print('Loaded ', len(self.trajectories), ' trajectories')
    
    def read_raw_data(self):
        return self.trajectories

# main
if __name__ == '__main__':
    # data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/data/nae_paper_dataset/data_preprocessed/Bamboo/data_train-134'
    # data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/data/nae_paper_dataset/origin/trimmed_Bamboo_168'
    # data_raw_reader = RoCatDataRawReader(data_dir)
    # data = data_raw_reader.read()

    data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/mocap_ws/src/mocap_data_collection/data/frisbee_ring/min_len_65/15-01-2025_10-42-46-traj_num-102.npz'
    thrown_object = 'ring_frisbee'
    data_raw_reader = RoCatRLLabDataRawReader(data_dir)
    data = data_raw_reader.read_raw_data()
    print('len data[0] keys: ', data[0].keys()) # ['points', 'orientations', 'msg_ids', 'time_stamps', 'low_freq_num']
    print('data[0]["points"]: ', data[0]['points'])
    time = data[0]['time_stamps']
    time = np.array(time) - time[0]
    for t in time:
        print(t)
    # for i in range(1, len(time)):
    #     if time[i] - time[1] < 1e-9:
    #         global_util_printer.print_red(f'time[{i}]: {time[i]} - time[0]: {time[0]}')
    #         input()
    # global_util_printer.print_green('Time is correct')
