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
from nae_core.utils.submodules.preprocess_utils.data_raw_reader import RoCatRLLabDataRawReader
from nae_core.utils.submodules.preprocess_utils.data_raw_correction_checker import RoCatRLDataRawCorrectionChecker
from nae_core.utils.submodules.preprocess_utils.data_splitter import RoCatDataSplitter
from python_utils.plotter import Plotter
from python_utils.printer import Printer
from scipy.interpolate import CubicSpline

global_util_plotter = Plotter()
global_util_printer = Printer()


# make a parent class for DataFormater
class DataFormater:
    def __init__(self):
        pass

    def format_data(self, data_folder, object_name='', save=False):
        pass
    
    def vel_interpolation(self, x_arr, t_arr, method='cubic'):
        '''
        Parameters:
            x_arr (numpy.ndarray): Position data.
            t_arr (numpy.ndarray): Time data (must be monotonically increasing).
            vel_arr (numpy.ndarray): Velocity data.
            method:
                - 'cubic': use CubicSpline to interpolate
                - 'linear': use np.gradient() to interpolate
                - 'manual': use manual interpolation
                    Interpolate velocities manually instead of using np.gradient() because np.gradient() auto interpolates if delta t between time steps are not equal.
                    Apply:
                        - Forward difference for the first point
                        - Backward difference for the last point
                        - Central difference for the rest points
        Returns:
            numpy.ndarray: Interpolated accelerations.
        '''

        if len(x_arr) != len(t_arr):
            raise ValueError("x_arr and t_arr must have the same length.")
        if not np.all(np.diff(t_arr) > 0):
            raise ValueError("t_arr must be monotonically increasing.")

        if method == 'cubic':
            # Tạo spline từ dữ liệu vị trí
            cs = CubicSpline(t_arr, x_arr)
            # Tính vận tốc và gia tốc từ spline
            vel = cs.derivative(1)(t_arr)  # Vận tốc
        elif method == 'linear':
            vel = np.gradient(x_arr, t_arr)
        elif method == 'manual':
            vel = self.vel_interpolation_manual(x_arr, t_arr)
        return vel
    
    def vel_interpolation_manual(self, x_arr, t_arr):
        '''
        method:
        - 'cubic': use CubicSpline to interpolate
        - 'linear': use np.gradient() to interpolate
        - 'manual': use manual interpolation
            Interpolate velocities manually instead of using np.gradient() because np.gradient() auto interpolates if delta t between time steps are not equal.
            Apply:
                - Forward difference for the first point
                - Backward difference for the last point
                - Central difference for the rest points
        '''
        vel = []  # Khởi tạo danh sách rỗng
        for i in range(len(x_arr)):
            if i == 0:
                # Forward difference
                vel_i = (x_arr[i + 1] - x_arr[i]) / (t_arr[i + 1] - t_arr[i])
            elif i == len(x_arr) - 1:
                # Backward difference
                vel_i = (x_arr[i] - x_arr[i - 1]) / (t_arr[i] - t_arr[i - 1])
            else:
                # Central difference
                vel_i = (x_arr[i + 1] - x_arr[i - 1]) / (t_arr[i + 1] - t_arr[i - 1])
            
            vel.append(vel_i)  # Thêm giá trị vào danh sách
        
        return np.array(vel)  # Trả về kết quả dưới dạng numpy array

    def acc_interpolation(self, x_arr, t_arr, vel_arr, method='cubic'):
        '''
        Parameters:
            x_arr (numpy.ndarray): Position data.
            t_arr (numpy.ndarray): Time data (must be monotonically increasing).
            vel_arr (numpy.ndarray): Velocity data.
            method:
                - 'cubic': use CubicSpline to interpolate
                - 'linear': use np.gradient() to interpolate
                - 'manual': use manual interpolation
                    Interpolate velocities manually instead of using np.gradient() because np.gradient() auto interpolates if delta t between time steps are not equal.
                    Apply:
                        - Forward difference for the first point
                        - Backward difference for the last point
                        - Central difference for the rest points
        Returns:
            numpy.ndarray: Interpolated accelerations.
        '''

        if len(x_arr) != len(t_arr):
            raise ValueError("x_arr and t_arr must have the same length.")
        if not np.all(np.diff(t_arr) > 0):
            raise ValueError("t_arr must be monotonically increasing.")
        
        if method == 'cubic':
            # Tạo spline từ dữ liệu vị trí
            cs = CubicSpline(t_arr, x_arr)
            # Tính vận tốc và gia tốc từ spline
            acc = cs.derivative(2)(t_arr)  # Gia tốc
        elif method == 'linear':
            # acc = np.gradient(vel_arr, t_arr)
            # 2nd derivative of position based on np.gradient() of position
            vel = np.gradient(x_arr, t_arr)  # Tính vận tốc
            acc = np.gradient(vel, t_arr)   # Tính gia tốc

        elif method == 'manual':
            acc = self.acc_interpolation_manual(x_arr, t_arr, vel_arr)

        return acc
    
    def acc_interpolation_manual(self, x_arr, t_arr, vel_arr, cal_type='2nd_derivative_pos'):
        """
        Interpolate accelerations manually using finite differences.
        Parameters:
            x_arr (numpy.ndarray): Position data.
            t_arr (numpy.ndarray): Time data (must be monotonically increasing).
            vel_arr (numpy.ndarray): Velocity data.
            cal_type (str): Type of acceleration calculation. Default is '2nd_derivative_pos'.
                - '2nd_derivative_pos': Calculate acceleration using 2nd derivative of position.
                - '1st_derivative_vel': Calculate acceleration using 1st derivative of velocity.
        Returns:
            numpy.ndarray: Interpolated accelerations.
        """
        if len(x_arr) != len(t_arr):
            raise ValueError("x_arr and t_arr must have the same length.")
        if not np.all(np.diff(t_arr) > 0):
            raise ValueError("t_arr must be monotonically increasing.")

        acc = np.zeros_like(x_arr)  # Initialize acceleration array with zeros

        for i in range(len(x_arr)):
            if i == 0:
                # Forward difference for the first point
                dt1 = t_arr[i + 1] - t_arr[i]
                if cal_type == '2nd_derivative_pos':
                    acc[i] = (x_arr[i + 1] - x_arr[i]) / (dt1 ** 2)
                elif cal_type == '1st_derivative_vel':
                    acc[i] = (vel_arr[i + 1] - vel_arr[i]) / dt1

            elif i == len(x_arr) - 1:
                # Backward difference for the last point
                dt1 = t_arr[i] - t_arr[i - 1]
                if cal_type == '2nd_derivative_pos':
                    acc[i] = (x_arr[i] - x_arr[i - 1]) / (dt1 ** 2)
                elif cal_type == '1st_derivative_vel':
                    acc[i] = (vel_arr[i] - vel_arr[i - 1]) / dt1
            else:
                # Central difference for second derivative
                dt1 = t_arr[i] - t_arr[i - 1]
                dt2 = t_arr[i + 1] - t_arr[i]
                if cal_type == '2nd_derivative_pos':
                    acc[i] = 2 * (x_arr[i + 1] - x_arr[i]) / (dt2 * (dt1 + dt2)) - \
                            2 * (x_arr[i] - x_arr[i - 1]) / (dt1 * (dt1 + dt2))
                elif cal_type == '1st_derivative_vel':
                    acc[i] = 2 * (vel_arr[i + 1] - vel_arr[i]) / (dt2 * (dt1 + dt2)) - \
                            2 * (vel_arr[i] - vel_arr[i - 1]) / (dt1 * (dt1 + dt2))
        return acc
    
    def save_info_txt(self, ):
        pass

class RLLabDatasetDataFormater(DataFormater):
    def __init__(self):
        pass

    '''
    seperate each trajectory in the dataset to a npz file
    '''
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
            
            # save this trajectory to npz file without interpolation
            new_d = {
                'position': np.array([pos[:, 0], pos[:, 1], pos[:, 2]]).T,
                'msg_ids': np.array(d_raw['msg_ids']),
                'time_stamps': np.array(time)
            }
            data_new.append(new_d)

        if save:
            for i, d in enumerate(data_new):
                self.save_every_traj_to_npz(d, object_name, i, len(pos))

        return data_new
    

    def save_every_traj_to_npz(self, new_d, object_name, data_idx, total_data_num):
        # save new_d to npz
        # get current path
        current_path = os.path.dirname(os.path.realpath(__file__))
        # get parent path
        parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
        # save new data
        if object_name == '':
            object_name = data_path.split('/')[-2]
        new_data_folder = os.path.join(parent_path, 'data', object_name)
        file_name = f'{object_name}_traj_{data_idx}_len_{total_data_num}_{data_idx}.npz'
        
        # mkdir if not exist
        if not os.path.exists(new_data_folder):
            os.makedirs(new_data_folder)
        new_data_path = os.path.join(new_data_folder, file_name)
        np.savez(new_data_path, position=new_d['position'], msg_ids=new_d['msg_ids'], time_stamps=new_d['time_stamps'])
        global_util_printer.print_green(f'Save trajectory {data_idx} to {new_data_path}')
    
    def save_info_txt(self, save_dir, original_data_path, data_raw, data_new, object_name, init_points_removed_num, last_points_removed_num, min_len_traj, too_short_traj_count, data_removed_idx_list, vel_interpolation_method, acc_interpolation_method):
        # make info.txt file
        with open(os.path.join(save_dir, 'info_data_format.txt'), 'w') as f:
            f.write('----- Data format info -----\n')
            f.write(f'Object name: {object_name}\n')
            f.write(f'Original data path: {original_data_path}\n')
            f.write(f'  Initial points removed: {init_points_removed_num}\n')
            f.write(f'  Last points removed: {last_points_removed_num}\n')
            f.write(f'  Data removed index: {data_removed_idx_list}\n')
            f.write(f'  min_len_traj: {min_len_traj}\n')
            f.write(f'  vel_interpolation_method: {vel_interpolation_method}\n')
            f.write(f'  acc_interpolation_method: {acc_interpolation_method}\n')
            f.write(f'  Data too short: {too_short_traj_count}\n')
            f.write('   Data raw:\n')
            f.write(f'      Trajectories: {len(data_raw)}\n')
            f.write('   Data new:\n')
            f.write(f'      Trajectories: {len(data_new)}\n')
            f.write('-------------------------------------')
            f.close()

if __name__ == '__main__':
    # data_path = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/data/new_dataset_no_orientation/new_format/1_original/big_plane_merged_264_ORIGIN.npz'
    # object_name = 'plane'

    data_path = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/data/new_dataset_no_orientation/new_format/1_original/frisbee-pbl_num_275_ORIGIN.npz'
    object_name = 'frisbee'
    data_removed_idx_list = []
    min_len_traj = None
    data_formater = RLLabDatasetDataFormater()
    new_data = data_formater.format_data(data_path, object_name=object_name, save=True, 
                                         init_points_removed_num=5, last_points_removed_num=2, min_len_traj=min_len_traj, data_removed_idx_list=data_removed_idx_list)

    print('After format data: ', len(new_data))

