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
from scipy.interpolate import UnivariateSpline, CubicSpline

global_util_plotter = Plotter()
global_util_printer = Printer()


# make a parent class for DataFormater
class DataFormater:
    def __init__(self):
        pass

    def format_data(self, data_folder, object_name='', save=False):
        pass
    
    def vel_interpolation(self, x_arr, t_arr, method='spline'):
        '''
        Parameters:
            x_arr (numpy.ndarray): Position data.
            t_arr (numpy.ndarray): Time data (must be monotonically increasing).
            vel_arr (numpy.ndarray): Velocity data.
            method:
                - 'spline': use UnivariateSpline to interpolate
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

        if method == 'spline':
            # Tạo spline từ dữ liệu vị trí
            spline = UnivariateSpline(t_arr, x_arr, k=3)
            # Tính vận tốc và gia tốc từ spline
            vel = spline.derivative(n=1)(t_arr)  # Vận tốc
        elif method == 'linear':
            vel = np.gradient(x_arr, t_arr)
        elif method == 'manual':
            vel = self.vel_interpolation_manual(x_arr, t_arr)
        return vel
    
    def vel_interpolation_manual(self, x_arr, t_arr):
        '''
        method:
        - 'spline': use UnivariateSpline to interpolate
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

    def acc_interpolation(self, x_arr, t_arr, vel_arr, method='spline'):
        '''
        Parameters:
            x_arr (numpy.ndarray): Position data.
            t_arr (numpy.ndarray): Time data (must be monotonically increasing).
            vel_arr (numpy.ndarray): Velocity data.
            method:
                - 'spline': use UnivariateSpline to interpolate
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
        
        # check if the method string has "spline" inside:
        if 'spline' in method:
            # tách method thành spline và bậc của spline từ string (check xem có đúng định dạng không: spline_number)
            method_split = method.split('-')
            if len(method_split) != 2 or int(method_split[1]) < 2:
                global_util_printer.print_yellow("Method must be in the format of 'spline-number' with number >= 2.")
                global_util_printer.print_yellow('Auto switch to CubicSpline')
                spline = CubicSpline(t_arr, x_arr)
            else:    
                spline_order = int(method_split[1])  # Lấy số bậc spline từ string
                # Tạo spline từ dữ liệu vị trí
                spline = UnivariateSpline(t_arr, x_arr, k=spline_order)
                # Tính vận tốc và gia tốc từ spline
            acc = spline.derivative(2)(t_arr)  # Gia tốc
        if method == 'linear':
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

    def x_vs_timestamp(self, x):
        t = []
        for i in range(x.shape[0]):
            t.append(i/120)
        t = np.array(t)
        return x, t

class NAEDatasetDataFormater(DataFormater):
    def __init__(self):
        pass

    def format_data(self, data_folder, object_name='', save=False, vel_interpolation_method='manual', acc_interpolation_method='manual', format_time=False):
        data_reader = RoCatNAEDataRawReader(data_folder)
        data = data_reader.read()
        print('Data field: ', data[0].files)      # ['frame_num', 'time_step', 'position', 'quaternion']
        data_new = []
        for d in data:
            pos = d['position']
            if format_time:
                time = self.x_vs_timestamp(pos[:, 0])[1]
            else:
                time = d['time_step']

            if pos.shape[0] != len(time):
                raise ValueError("Position data and time steps must have the same length.")
            vel_x = self.vel_interpolation(pos[:, 0], time, method=vel_interpolation_method)     # vel_x = d_x/d_t
            vel_y = self.vel_interpolation(pos[:, 1], time, method=vel_interpolation_method)     # vel_y = d_y/d_t
            vel_z = self.vel_interpolation(pos[:, 2], time, method=vel_interpolation_method)     # vel_z = d_z/d_t

            acc_x = self.acc_interpolation(pos[:, 0], time, vel_x, method=acc_interpolation_method)     # acc_x = d^2_x/d_t^2
            acc_y = self.acc_interpolation(pos[:, 1], time, vel_y, method=acc_interpolation_method)     # acc_y = d^2_y/d_t^2
            acc_z = self.acc_interpolation(pos[:, 2], time, vel_z, method=acc_interpolation_method)     # acc_z = d^2_z/d_t^2

            # create new data with the new format
            new_d = {
                'points': np.array([pos[:, 0], pos[:, 1], pos[:, 2], vel_x, vel_y, vel_z, acc_x, acc_y, acc_z]).T,
                'msg_ids': np.array(d['frame_num']),
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
                object_name = data_folder.split('/')[-1]
            new_data_folder = os.path.join(parent_path, 'data', object_name)
            if not os.path.exists(new_data_folder):
                os.makedirs(new_data_folder)
            new_data_path = os.path.join(new_data_folder, object_name + '_new_data_format_num_' + str(len(data_new)) + '.npz')
            np.savez(new_data_path, **new_data_dict)
            global_util_printer.print_green(f'Saved new data to: {new_data_path}')

        # print new_data_dict info
        print('New data info:')
        print('     - new_data_dict.keys(): ', new_data_dict.keys())
        print('     - new_data_dict[trajectories]: ', len(new_data_dict['trajectories']))
        print('     - new_data_dict[object_name]: ', new_data_dict['object_name'])
        print('     - new_data_dict[trajectories][0].keys(): ', new_data_dict['trajectories'][0].keys())
        print('     - new_data_dict[trajectories][0][points]: ', new_data_dict['trajectories'][0]['points'].shape)
        print('-------------------------------------')

        self.save_info_txt(new_data_folder, data_folder, data, data_new, object_name, vel_interpolation_method, acc_interpolation_method)
        return data_new
    
    def save_info_txt(self, save_dir, original_data_path, data_raw, data_new, object_name, vel_interpolation_method, acc_interpolation_method):
        # make info.txt file
        with open(os.path.join(save_dir, 'info_data_format.txt'), 'w') as f:
            f.write('----- Data format info -----\n')
            f.write(f'Object name: {object_name}\n')
            f.write(f'Original data path: {original_data_path}\n')
            f.write(f'  vel_interpolation_method: {vel_interpolation_method}\n')
            f.write(f'  acc_interpolation_method: {acc_interpolation_method}\n')
            f.write('   Data raw:\n')
            f.write(f'      Trajectories: {len(data_raw)}\n')
            f.write('   Data new:\n')
            f.write(f'      Trajectories: {len(data_new)}\n')
            f.write('-------------------------------------')
            f.close()
    
class RLLabDatasetDataFormater(DataFormater):
    def __init__(self):
        pass

    def format_data(self, data_path, object_name='', save=False, init_points_removed_num=0, last_points_removed_num=0, min_len_traj=None, data_removed_idx_list=[], \
                    vel_interpolation_method='manual', acc_interpolation_method='manual', format_time=False):
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
            if format_time:
                time = self.x_vs_timestamp(pos[:, 0])[1]
            else:
                time = d_raw['time_stamps']
            vel_x = self.vel_interpolation(pos[:, 0], time, method=vel_interpolation_method)     # vel_x = d_x/d_t
            vel_y = self.vel_interpolation(pos[:, 1], time, method=vel_interpolation_method)     # vel_y = d_y/d_t
            vel_z = self.vel_interpolation(pos[:, 2], time, method=vel_interpolation_method)     # vel_z = d_z/d_t
            acc_x = self.acc_interpolation(pos[:, 0], time, vel_x, method=acc_interpolation_method)     # acc_x = d^2_x/d_t^2
            acc_y = self.acc_interpolation(pos[:, 1], time, vel_y, method=acc_interpolation_method)     # acc_y = d^2_y/d_t^2
            acc_z = self.acc_interpolation(pos[:, 2], time, vel_z, method=acc_interpolation_method)     # acc_z = d^2_z/d_t^2

            # create new data with the new format
            new_d = {
                # 'points': np.array([pos[:, 0], pos[:, 1], pos[:, 2], vel_x, vel_y, vel_z, np.zeros(len(time)), -9.81*np.ones(len(time)), np.zeros(len(time))]).T,
                'points': np.array([pos[:, 0], pos[:, 1], pos[:, 2], vel_x, vel_y, vel_z, acc_x, acc_y, acc_z]).T,
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
            self.save_info_txt(new_data_folder, data_path, data_raw, data_new, object_name, init_points_removed_num, last_points_removed_num, min_len_traj, too_short_traj_count, data_removed_idx_list, vel_interpolation_method, acc_interpolation_method)
            global_util_printer.print_green(f'Saved new data to: {new_data_path}')
        
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
    # # =========================== RLLab dataset format ===========================
    # data_path = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/data/new_dataset_no_orientation/new_format/big_plane_merged_264_ORIGIN.npz'
    # object_name = 'big_plane'
    # data_removed_idx_list = []
    # min_len_traj = 70

    # data_path = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/data/new_dataset_no_orientation/new_format/frisbee-pbl_num_275_ORIGIN.npz'
    # object_name = 'frisbee'
    # data_removed_idx_list = [35, 80, 84, 243, 261, 158] # only apply for frisbee
    # min_len_traj = 70

    # data_path = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/data/new_dataset_no_orientation/new_format/boomerang_num-252_ORIGIN.npz'
    # object_name = 'boomerang'
    # data_removed_idx_list = []
    # min_len_traj = 70

    # =========================== NAE dataset format ===========================
    data_path = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/data/nae_paper_dataset/origin/trimmed_Bottle_115'
    object_name = 'bottle'
    format_time = False

    data_formater = NAEDatasetDataFormater()
    vel_interpolation_method = 'manual'      # 'spline', 'linear', 'manual'
    acc_interpolation_method = 'spline-3'      # 'spline', 'linear', 'manual'
    new_data = data_formater.format_data(data_path, object_name=object_name, save=True, \
                                         vel_interpolation_method=vel_interpolation_method, acc_interpolation_method=acc_interpolation_method, format_time=format_time)

    # # =========================== RLLab dataset format ===========================
    # # data_path = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/data/new_dataset_no_orientation/new_format/1_original/big_plane_merged_264_ORIGIN.npz'
    # # object_name = 'big_plane'

    # data_path = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/data/new_dataset_no_orientation/new_format/1_original/boomerang_num-252_ORIGIN.npz'
    # object_name = 'boomerang'
    # data_removed_idx_list = []
    # min_len_traj = None
    # data_formater = RLLabDatasetDataFormater()
    # vel_interpolation_method = 'manual'      # 'spline', 'linear', 'manual'
    # acc_interpolation_method = 'spline'      # 'spline', 'linear', 'manual'
    # format_time = False
    # new_data = data_formater.format_data(data_path, object_name=object_name, save=True, 
    #                                      init_points_removed_num=5, last_points_removed_num=2, min_len_traj=min_len_traj, data_removed_idx_list=data_removed_idx_list,
    #                                      vel_interpolation_method=vel_interpolation_method,
    #                                      acc_interpolation_method=acc_interpolation_method)

    # plot acceleration of the first trajectory
    acc_x = new_data[0]['points'][:, 6]
    acc_y = new_data[0]['points'][:, 7]
    acc_z = new_data[0]['points'][:, 8]
    if format_time:
        time = data_formater.x_vs_timestamp(acc_x)[1]
    else:
        time = new_data[0]['time_stamps']
    global_util_plotter.plot_line_chart(x_values=time, y_values=[acc_x, acc_y, acc_z], \
                                        title=f'111111 Acceleration - {object_name}  |  VEL_interpolate: {vel_interpolation_method} - ACC_interpolate: {acc_interpolation_method} - Fomat_time: {format_time}', \
                                        x_label='Time step', y_label='Acceleration x', \
                                        legends=['acc_x', 'acc_y', 'acc_z'])

    # # plot velocity of the first trajectory
    # vel_x = new_data[0]['points'][:, 3]
    # vel_y = new_data[0]['points'][:, 4]
    # vel_z = new_data[0]['points'][:, 5]
    # x_values = np.arange(len(vel_x))
    # global_util_plotter.plot_line_chart(x_values=x_values, y_values=[vel_x, vel_y, vel_z], \
    #                                     title=f'Velocity - {object_name}  |  VEL_interpolate: {vel_interpolation_method} - ACC_interpolate: {acc_interpolation_method}', \
    #                                     x_label='Time step', y_label='Velocity', \
    #                                     legends=['vel_x', 'vel_y', 'vel_z'])

    # # plot position of the first trajectory
    # pos_x = new_data[0]['points'][:, 0]
    # pos_y = new_data[0]['points'][:, 1]
    # pos_z = new_data[0]['points'][:, 2]
    # x_values = np.arange(len(pos_x))
    # global_util_plotter.plot_line_chart(x_values=x_values, y_values=[pos_x, pos_y, pos_z], title=f'Position - {object_name} - {interpolation_method}', x_label='Time step', y_label='Position z')


    # input('Do you want to check data correction?')
    # data_correction_checker = RoCatRLDataRawCorrectionChecker()
    # data_correction_checker.check_data_correction(new_data, data_whose_y_up=True)

    input('Do you want to split data?')
    data_splitter = RoCatDataSplitter(new_data, train_ratio=0.8, val_ratio=0.1, num_first_points_to_cut=0, object_name=object_name)
    data_splitter.split(shuffle_data=True) 

