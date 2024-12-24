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
from nae_core.utils.submodules.preprocess_utils.data_raw_reader import RoCatRLLabDataRawReader, RoCatNAEDataRawReader
from nae_core.utils.submodules.preprocess_utils.data_raw_correction_checker import RoCatRLDataRawCorrectionChecker
from nae_core.utils.submodules.preprocess_utils.data_splitter import RoCatDataSplitter
from python_utils.plotter import Plotter
from python_utils.printer import Printer
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

global_util_plotter = Plotter()
global_util_printer = Printer()

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
            new_data_folder = os.path.join(parent_path, 'data', object_name)
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

    def format_data(self, data_path, object_name='', save=False, init_points_removed_num=0, last_points_removed_num=0, min_len_traj=None, data_removed_idx_list=[]):
        data_reader = RoCatRLLabDataRawReader(data_path)
        data_raw = data_reader.read_raw_data()
        data_new = []
        too_short_traj_count = 0

        for idx, d_raw in enumerate(data_raw):
            # filter outliers manually
            if idx in data_removed_idx_list:
                global_util_printer.print_yellow(f'Ignore outlier trajectory {idx}')
                # global_util_plotter.plot_trajectory_dataset_matplotlib([d['points']], title=f'Trajectory {idx}', rotate_data_whose_y_up=True, plot_all=False, shuffle=False)
                continue
            
            # dict_keys(['points', 'msg_ids', 'time_stamps', 'low_freq_num'])
            # remove initial points
            if init_points_removed_num > 0:
                d_raw['points'] = d_raw['points'][init_points_removed_num:]
                d_raw['msg_ids'] = d_raw['msg_ids'][init_points_removed_num:]
                d_raw['time_stamps'] = d_raw['time_stamps'][init_points_removed_num:]
            # remove last points
            if last_points_removed_num > 0:
                d_raw['points'] = d_raw['points'][:-last_points_removed_num]
                d_raw['msg_ids'] = d_raw['msg_ids'][:-last_points_removed_num]
                d_raw['time_stamps'] = d_raw['time_stamps'][:-last_points_removed_num]
            
            # remove trajectory with length < min_len_traj
            if min_len_traj is not None:
                if len(d_raw['points']) < min_len_traj:
                    global_util_printer.print_yellow(f'Ignore trajectory {idx} with length {len(d_raw["points"])} < {min_len_traj}')
                    too_short_traj_count += 1
                    continue
            
            pos = d_raw['points']
            time = d_raw['time_stamps']
            vel_x = self.vel_interpolation(pos[:, 0], time)     # vel_x = d_x/d_t
            vel_y = self.vel_interpolation(pos[:, 1], time)     # vel_y = d_y/d_t
            vel_z = self.vel_interpolation(pos[:, 2], time)     # vel_z = d_z/d_t

            # create new data with the new format
            new_d = {
                'points': np.array([pos[:, 0], pos[:, 1], pos[:, 2], vel_x, vel_y, vel_z, np.zeros(len(time)), -9.81*np.ones(len(time)), np.zeros(len(time))]).T,
                'msg_ids': np.array(d_raw['msg_ids']),
                'time_stamps': np.array(time)
            }
            data_new.append(new_d)

        if save:
            new_data_dict = {'trajectories': data_new,
                            'object_name': object_name}
            # get current path
            current_path = os.path.dirname(os.path.realpath(__file__))
            # get parent path
            parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
            # save new data
            if object_name == '':
                object_name = data_path.split('/')[-2]
            new_data_folder = os.path.join(parent_path, 'data', object_name)
            if not os.path.exists(new_data_folder):
                os.makedirs(new_data_folder)
            new_data_path = os.path.join(new_data_folder, object_name + '_new_data_format_num_' + str(len(data_new)) + '.npz')
            np.savez(new_data_path, **new_data_dict)
            self.save_info_txt(new_data_folder, data_path, data_raw, data_new, object_name, init_points_removed_num, last_points_removed_num, min_len_traj, too_short_traj_count, data_removed_idx_list)
            print('Saved new data to: ', new_data_path)
        
        # print new_data_dict info
        print('----- Data format info -----')
        print(f'Object name: {object_name}')
        print(f'  Initial points removed: {init_points_removed_num}')
        print(f'  Last points removed: {last_points_removed_num}')
        print(f'  Data removed index: {data_removed_idx_list}')
        print(f'  min_len_traj: {min_len_traj}')
        print(f'  Data too short: {too_short_traj_count}')
        print('   Data raw:')
        print(f'      Trajectories: {len(data_raw)}')
        print('   Data new:')
        print(f'      Trajectories: {len(data_new)}')
        print('-------------------------------------')

        return data_new
    
    def save_info_txt(self, save_dir, original_data_path, data_raw, data_new, object_name, init_points_removed_num, last_points_removed_num, min_len_traj, too_short_traj_count, data_removed_idx_list):
        # make info.txt file
        with open(os.path.join(save_dir, 'info_data_format.txt'), 'w') as f:
            f.write('----- Data format info -----\n')
            f.write(f'Object name: {object_name}\n')
            f.write(f'Original data path: {original_data_path}\n')
            f.write(f'  Initial points removed: {init_points_removed_num}\n')
            f.write(f'  Last points removed: {last_points_removed_num}\n')
            f.write(f'  Data removed index: {data_removed_idx_list}\n')
            f.write(f'  min_len_traj: {min_len_traj}\n')
            f.write(f'  Data too short: {too_short_traj_count}\n')
            f.write('   Data raw:\n')
            f.write(f'      Trajectories: {len(data_raw)}\n')
            f.write('   Data new:\n')
            f.write(f'      Trajectories: {len(data_new)}\n')
            f.write('-------------------------------------')
            f.close()

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


if __name__ == '__main__':
    # =========================== RLLab dataset format ===========================
    data_path = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/data/new_dataset_no_orientation/new_format/big_plane_merged_264_ORIGIN.npz'
    object_name = 'big_plane'
    data_removed_idx_list = []
    min_len_traj = 70

    # data_path = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/data/new_dataset_no_orientation/new_format/frisbee-pbl_num_275_ORIGIN.npz'
    # object_name = 'frisbee'
    # data_removed_idx_list = [35, 80, 84, 243, 261, 158] # only apply for frisbee
    # min_len_traj = 70

    # data_path = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/data/new_dataset_no_orientation/new_format/boomerang_num-252_ORIGIN.npz'
    # object_name = 'boomerang'
    # data_removed_idx_list = []
    # min_len_traj = 70



    global_util_printer.print_blue(f'Formating {object_name} data...', background=True)
    data_formater = RLLabDatasetDataFormater()
    global_util_printer.print_yellow('Are you sure to remove the following data points: ' + str(data_removed_idx_list))
    data_removed_idx_list_answer = input()
    if data_removed_idx_list_answer == 'n':
        data_removed_idx_list = []
    elif data_removed_idx_list_answer == 'y':
        pass
    else:
        raise ValueError('Invalid input')
    new_data = data_formater.format_data(data_path, object_name=object_name, save=True, init_points_removed_num=5, last_points_removed_num=2, min_len_traj=min_len_traj, data_removed_idx_list=data_removed_idx_list)

    input('Do you want to check data correction?')
    data_correction_checker = RoCatRLDataRawCorrectionChecker()
    data_correction_checker.check_data_correction(new_data, data_whose_y_up=True)

    input('Do you want to split data?')
    data_splitter = RoCatDataSplitter(new_data, train_ratio=0.8, val_ratio=0.1, num_first_points_to_cut=0, object_name=object_name)
    data_splitter.split(shuffle_data=True) 
