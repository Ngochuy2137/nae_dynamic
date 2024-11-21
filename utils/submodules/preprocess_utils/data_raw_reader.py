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

class RoCatNAEDataRawReader:
    def __init__(self, file_path):
        self.trajectories = self.load_data(file_path)
        print('Loaded ', len(self.trajectories), ' trajectories')
        # print('Data field: ', self.trajectories[0].files)      # ['frame_num', 'time_step', 'position', 'quaternion']
    
    def load_data(self, data_folder):
        # load all npz data files
        # Find all npz files in the folder
        npz_files = [f for f in os.listdir(data_folder) if f.endswith('.npz')]
        print('Found ', len(npz_files), ' npz files in ', data_folder)
        data = []
        for f in npz_files:
            file_path = os.path.join(data_folder, f)
            traj = np.load(file_path, allow_pickle=True)
            data.append(traj)
        return data
    
    def read(self):
        return self.trajectories
    
class RoCatRLLabDataRawReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.trajectories = np.load(self.file_path, allow_pickle=True)
        print('Loaded file with ', len(self.trajectories['trajectories']), ' trajectories')

        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Trajectory of the object')
        # self.ax.set_box_aspect([1,1,1])
        self.ax.view_init(elev=20, azim=30)

    def read(self):
        return self.trajectories['trajectories']
    
# main
if __name__ == '__main__':
    ## ========================= RLLAB dataset =========================
    # data_reader = RoCatRLLabDataRawReader('/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_prediction_ws/src/nae/data/new_dataset_no_orientation/frisbee/frisbee-pbl/frisbee-pbl_merged_275.npz')
    # data = data_reader.read()
    # # check all keys of the data[0]
    # # data_plot = [[d['points'], 'o'] for d in data]
    # data_plot = [d['points'] for d in data]
    # util_plotter = Plotter()

    # print('check data_plot:', len(data_plot))
    # util_plotter.plot_samples(data_plot)

    ## ========================= NAE dataset =========================
    data_folder = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_prediction_ws/src/nae/data/nae_paper_dataset/origin/trimmed_Bamboo_168'
    data_reader = RoCatNAEDataRawReader(data_folder)