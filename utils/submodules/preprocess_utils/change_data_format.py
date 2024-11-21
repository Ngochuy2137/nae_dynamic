'''
Change data format:

Old data format:
    Is a dictionary with keys: 'trajectories'
    Each data point of the old data format is:
    - Swap y, z
    - Swap vy, vz
    - Data point: x, y, z, vx, vy, vz, 0, 0, 9.81

New data format:
    Is a dictionary with keys: 'trajectories', 'object_name'
    Each data point of the new data format is:
    - Keep y, z, no swap anymore => need to:
        + Swap y, z
        + Swap vy, vz
    - Data point: x, y, z, vx, vy, vz, 0, -9.81, 0
'''

import numpy as np
import os
from nae.utils.submodules.preprocess_utils.data_raw_reader import RoCatRLLabDataRawReader, RoCatNAEDataRawReader
from nae.utils.submodules.preprocess_utils.data_raw_correction_checker import RoCatRLDataRawCorrectionChecker
from nae.utils.submodules.preprocess_utils.data_splitter import RoCatDataSplitter

'''
Format data to train NAE model
- This script includes 2 classes:
    + NAEDatasetDataFormater: format data for NAE dataset
    + RLLabDatasetDataFormater: format data for RLLab dataset
- The main function of each class is format_data(data_folder, object_name='', save=False)
- The target data format is:
    + Is a dictionary with keys: 'trajectories', 'object_name'
    + Each data point of each trajectory with the data format is:
        - x, y, z, vx, vy, vz, 0, -9.81, 0
        - Based on the data collected with mocap system (both NAE dataset and RLLab dataset), y axis is the up axis, and x, y, z follow the right-hand rule

'''
class NAEDatasetDataFormater:
    def __init__(self):
        pass

    def format_data(self, data_folder, object_name='', save=False):
        data_reader = RoCatNAEDataRawReader(data_folder)
        data = data_reader.read()
        print('Data field: ', data[0].files)      # ['frame_num', 'time_step', 'position', 'quaternion']
        new_trajectories = []
        for d in data:
            pos = d['position']
            time = d['time_step']
            vel_x = self.vel_interpolation(pos[:, 0], time)     # vel_x = d_x/d_t
            vel_y = self.vel_interpolation(pos[:, 1], time)     # vel_y = d_y/d_t
            vel_z = self.vel_interpolation(pos[:, 2], time)     # vel_z = d_z/d_t

            # create new data with the new format
            new_d = {
                'points': np.array([pos[:, 0], pos[:, 1], pos[:, 2], vel_x, vel_y, vel_z, np.zeros(len(time)), -9.81*np.ones(len(time)), np.zeros(len(time))]).T,
                'msg_ids': np.array(d['frame_num']),
                'time_stamps': np.array(time)
            }
            new_trajectories.append(new_d)
        if save:
            new_data_dict = {'trajectories': new_trajectories,
                            'object_name': object_name}
            # get current path
            current_path = os.path.dirname(os.path.realpath(__file__))
            # get parent path
            parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
            # save new data
            if object_name == '':
                object_name = data_folder.split('/')[-1]
            new_data_folder = os.path.join(parent_path, 'data')
            if not os.path.exists(new_data_folder):
                os.makedirs(new_data_folder)
            new_data_path = os.path.join(new_data_folder, object_name + '_new_data_format_num_' + str(len(new_trajectories)) + '.npz')
            np.savez(new_data_path, **new_data_dict)
            print('Saved new data to: ', new_data_path)

        # print new_data_dict info
        print('New data info:')
        print('     - new_data_dict.keys(): ', new_data_dict.keys())
        print('     - new_data_dict[trajectories]: ', len(new_data_dict['trajectories']))
        print('     - new_data_dict[object_name]: ', new_data_dict['object_name'])
        print('     - new_data_dict[trajectories][0].keys(): ', new_data_dict['trajectories'][0].keys())
        print('     - new_data_dict[trajectories][0][points]: ', new_data_dict['trajectories'][0]['points'].shape)
        print('-------------------------------------')

        return new_data_path



    def vel_interpolation(self, x_arr, t_arr):
        '''
        Interpolate velocities manually instead of using np.gradient() because np.gradient() auto interpolates if delta t between time steps are not equal.
        Apply:
            - Forward difference for the first point
            - Backward difference for the last point
            - Central difference for the rest points
        '''
        vel = []
        for i in range(len(x_arr)):
            if i == 0:
                # Forward difference
                prev_t_id = i
                next_t_id = i+1
            elif i == len(x_arr) - 1:
                # Backward difference
                prev_t_id = i-1
                next_t_id = i
            else:
                # Central difference
                prev_t_id = i-1
                next_t_id = i+1
            
            #check 0 division
            if t_arr[next_t_id] - t_arr[prev_t_id] == 0:
                vel_i = 0
            else:
                vel_i = (x_arr[next_t_id] - x_arr[prev_t_id]) / (t_arr[next_t_id] - t_arr[prev_t_id])
            vel.append(vel_i)
        vel = np.array(vel)
        return vel
    
class RLLabDatasetDataFormater:
    def __init__(self):
        pass

    def format_data(self, data_path, object_name='', save=False):
        data_reader = RoCatRLLabDataRawReader(data_path)
        data = data_reader.read()
        new_data = []
        for d in data:
            new_d = d.copy()
            old_points = d['points']
            new_points = []
            for po in old_points:
                new_po = po.copy()
                new_po[1], new_po[2] = new_po[2], new_po[1]
                new_po[4], new_po[5] = new_po[5], new_po[4]
                new_po[7] = -9.81
                new_po[8] = 0
                new_points.append(new_po)
            new_d['points'] = np.array(new_points)

            # print('new_d.keys(): ', new_d.keys())
            # print('check new_d[points]: ', new_d['points'].shape)
            # print('check new_d[msg_ids]: ', new_d['msg_ids'].shape)
            # print('check new_d[time_stamps]: ', new_d['time_stamps'].shape)
            # print('check new_d[low_freq_num]: ', new_d['low_freq_num'])
            # input()
            new_data.append(new_d)
        if save:
            new_data_dict = {'trajectories': new_data,
                            'object_name': object_name}
            # get current path
            current_path = os.path.dirname(os.path.realpath(__file__))
            # get parent path
            parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
            # save new data
            if object_name == '':
                object_name = data_path.split('/')[-2]
            new_data_folder = os.path.join(parent_path, 'data')
            if not os.path.exists(new_data_folder):
                os.makedirs(new_data_folder)
            new_data_path = os.path.join(new_data_folder, object_name + '_new_data_format_num_' + str(len(new_data)) + '.npz')
            np.savez(new_data_path, **new_data_dict)
            print('Saved new data to: ', new_data_path)
        
        # print new_data_dict info
        print('New data info:')
        print('     - new_data_dict.keys(): ', new_data_dict.keys())
        print('     - new_data_dict[trajectories]: ', len(new_data_dict['trajectories']))
        print('     - new_data_dict[object_name]: ', new_data_dict['object_name'])
        print('     - new_data_dict[trajectories][0].keys(): ', new_data_dict['trajectories'][0].keys())
        print('     - new_data_dict[trajectories][0][points]: ', new_data_dict['trajectories'][0]['points'].shape)
        print('-------------------------------------')

        return new_data


if __name__ == '__main__':

    ## =========================== RLLab dataset format ===========================
    # # data_path = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_prediction_ws/src/nae/data/new_dataset_no_orientation/frisbee/frisbee-pbl/frisbee-pbl_merged_275.npz'
    # # object_name = 'frisbee-pbl'

    # # data_path = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_prediction_ws/src/nae/data/new_dataset_no_orientation/boomerang/min_len_65/10-11-2024_17-06-36-traj_num-252.npz'
    # # object_name = 'boomerang'

    # data_path = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_prediction_ws/src/nae/data/new_dataset_no_orientation/big_plane/min_len_65/big_plane_merged_264.npz'
    # object_name = 'big_plane'
    # data_formater = RLLabDatasetDataFormater()
    # new_data = data_formater.format_data(data_path, object_name=object_name, save=True)

    # input('Do you want to check data correction?')
    # data_correction_checker = RoCatRLDataRawCorrectionChecker()
    # data_correction_checker.check_data_correction(new_data, new_data_format=True)

    # input('Do you want to split data?')
    # data_splitter = RoCatDataSplitter(new_data, train_ratio=0.8, val_ratio=0.1, num_first_points_to_cut=0, object_name=object_name)
    # data_splitter.split(shuffle_data=True) 

    ## =========================== NAE dataset format ===========================
    # data_folder = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_prediction_ws/src/nae/data/nae_paper_dataset/origin/trimmed_Bamboo_168'
    # object_name = 'bamboo'
    
    # data_folder = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_prediction_ws/src/nae/data/nae_paper_dataset/origin/trimmed_Banana_731'
    # object_name = 'banana'

    # data_folder = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_prediction_ws/src/nae/data/nae_paper_dataset/origin/trimmed_Bottle_115'
    # object_name = 'bottle'

    # data_folder = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_prediction_ws/src/nae/data/nae_paper_dataset/origin/trimmed_Gourd_203'
    # object_name = 'gourd'

    # data_folder = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_prediction_ws/src/nae/data/nae_paper_dataset/origin/trimmed_Green_139'
    # object_name = 'green'

    data_folder = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_prediction_ws/src/nae/data/nae_paper_dataset/origin/trimmed_Paige_171'
    object_name = 'paige'

    data_formater = NAEDatasetDataFormater()
    new_data_path = data_formater.format_data(data_folder, object_name=object_name, save=True)

    # check correction
    input('Do you want to check data correction?')
    data_reader = RoCatRLLabDataRawReader(new_data_path)
    new_data = data_reader.read()
    data_correction_checker = RoCatRLDataRawCorrectionChecker()
    data_correction_checker.check_data_correction(new_data, new_data_format=True)

    input('Do you want to split data?')
    data_splitter = RoCatDataSplitter(new_data, train_ratio=0.8, val_ratio=0.1, num_first_points_to_cut=0, object_name=object_name)
    data_splitter.split(shuffle_data=True)
