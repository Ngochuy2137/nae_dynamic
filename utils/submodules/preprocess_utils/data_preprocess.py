import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from python_utils.plotter import Plotter
from python_utils.printer import Printer
from scipy.interpolate import UnivariateSpline, CubicSpline
import os
import re
from sklearn.preprocessing import MinMaxScaler
import joblib  # Để lưu scaler
from tqdm import tqdm
from collections import defaultdict
import copy
import random
import torch

from nae_core.utils.submodules.preprocess_utils.data_raw_reader import RoCatDataRawReader

global_util_plotter = Plotter()
global_util_printer = Printer()

class DataNormalizer:
    def __init__(self, scaler_path=None, range_norm=(-0.5, 0.5)):
        global_util_printer.print_green("Constructing DataNormalizer...")
        if scaler_path is not None:
            self.scaler = joblib.load(scaler_path)
        elif range_norm is not None:
            self.scaler = MinMaxScaler(feature_range=range_norm)
        else:
            raise ValueError("Bạn cần cung cấp scaler_path hoặc range_norm.")
        self.range_norm = range_norm

    def normalize_data(self, data, object_name, note):
        if self.scaler is None:
            raise ValueError("Scaler chưa được khởi tạo.")
        data_normed = self.scaler.fit_transform(data)
        save_scaler = input('Do you want to save the scaler? (y/n): ')
        if save_scaler == 'y':
            self.save_scaler(object_name, note)
        return data_normed
    
    # def denormalize_data(self, data_normalized):
    #     if self.scaler is None:
    #         raise ValueError("Scaler chưa được khởi tạo.")
    #     return self.scaler.inverse_transform(data_normalized)

    def save_scaler(self, object_name, note):
        # get current path
        current_path = os.path.dirname(os.path.realpath(__file__))
        # get parent path
        parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
        parent_path = os.path.join(parent_path, 'data_preprocessed', object_name, 'normalization')
        # create folder if not exist
        if not os.path.exists(parent_path):
            os.makedirs(parent_path)
        # save new data
        scaler_path = os.path.join(parent_path, f'{note}_scaler.save')
        if self.scaler is None:
            raise ValueError("Scaler chưa được khởi tạo.")
        joblib.dump(self.scaler, scaler_path)
        global_util_printer.print_green(f'Saved scaler to {scaler_path}')

    def load_scaler(self, path):
        self.scaler = joblib.load(path)

class DataDenormalizer():
    def __init__(self, pos_scaler_path, vel_scaler_path, acc_scaler_path):
        self.denormalize_load_scaler(pos_scaler_path, vel_scaler_path, acc_scaler_path)

    def denormalize_load_scaler(self, pos_scaler_path=None, vel_scaler_path=None, acc_scaler_path=None):
        """
        Load MinMaxScaler parameters for position, velocity, and acceleration, and prepare tensors for GPU.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load scalers
        pos_scaler = joblib.load(pos_scaler_path) if pos_scaler_path else None
        vel_scaler = joblib.load(vel_scaler_path) if vel_scaler_path else None
        acc_scaler = joblib.load(acc_scaler_path) if acc_scaler_path else None

        # Handle position scaler
        if pos_scaler:
            self.min_pos = torch.tensor(pos_scaler.data_min_, device=device).view(1, 1, -1)
            self.max_pos = torch.tensor(pos_scaler.data_max_, device=device).view(1, 1, -1)
            print("Position MinMaxScaler parameters loaded.")

        # Handle velocity scaler
        if vel_scaler:
            self.min_vel = torch.tensor(vel_scaler.data_min_, device=device).view(1, 1, -1)
            self.max_vel = torch.tensor(vel_scaler.data_max_, device=device).view(1, 1, -1)
            print("Velocity MinMaxScaler parameters loaded.")

        # Handle acceleration scaler
        if acc_scaler:
            self.min_acc = torch.tensor(acc_scaler.data_min_, device=device).view(1, 1, -1)
            self.max_acc = torch.tensor(acc_scaler.data_max_, device=device).view(1, 1, -1)
            print("Acceleration MinMaxScaler parameters loaded.")

    def denormalize_data_on_cuda(self, data, denorm_pos=True, denorm_vel=True, denorm_acc=True):
        """
        Denormalize position, velocity, and acceleration data using preloaded tensors.
        Args:
            data (torch.Tensor): Normalized data (shape: [batch_size, time_steps, feature]).
            denorm_pos (bool): If True, denormalize position data.
            denorm_vel (bool): If True, denormalize velocity data.
            denorm_acc (bool): If True, denormalize acceleration data.
        Returns:
            torch.Tensor: Denormalized data (shape: [batch_size, time_steps, feature]).
        """
        denormalized_data = data.clone()  # Clone the input tensor to avoid modifying it directly

        # Denormalize position (x, y, z)
        if denorm_pos:
            print("Denormalizing position data...")
            denormalized_data[:, :, 0:3] = denormalized_data[:, :, 0:3] * (self.max_pos - self.min_pos) + self.min_pos

        # Denormalize velocity (vx, vy, vz)
        if denorm_vel:
            print("Denormalizing velocity data...")
            denormalized_data[:, :, 3:6] = denormalized_data[:, :, 3:6] * (self.max_vel - self.min_vel) + self.min_vel

        # Denormalize acceleration (ax, ay, az)
        if denorm_acc:
            print("Denormalizing acceleration data...")
            denormalized_data[:, :, 6:9] = denormalized_data[:, :, 6:9] * (self.max_acc - self.min_acc) + self.min_acc

        return denormalized_data
    
    def denormalize_data_all_feature_on_cuda(self, data):
        """
        Denormalize position, velocity, and acceleration data using preloaded tensors.
        Args:
            data (torch.Tensor): Normalized data (shape: [batch_size, time_steps, feature]).
            denorm_pos (bool): If True, denormalize position data.
            denorm_vel (bool): If True, denormalize velocity data.
            denorm_acc (bool): If True, denormalize acceleration data.
        Returns:
            torch.Tensor: Denormalized data (shape: [batch_size, time_steps, feature]).
        """
        denormalized_data = data.clone()  # Clone the input tensor to avoid modifying it directly

        # Denormalize position (x, y, z)
        print("Denormalizing position data...")
        denormalized_data[:, :, 0:3] = denormalized_data[:, :, 0:3] * (self.max_pos - self.min_pos) + self.min_pos

        # Denormalize velocity (vx, vy, vz)
        print("Denormalizing velocity data...")
        denormalized_data[:, :, 3:6] = denormalized_data[:, :, 3:6] * (self.max_vel - self.min_vel) + self.min_vel

        # Denormalize acceleration (ax, ay, az)
        print("Denormalizing acceleration data...")
        denormalized_data[:, :, 6:9] = denormalized_data[:, :, 6:9] * (self.max_acc - self.min_acc) + self.min_acc

        return denormalized_data

class DataPreprocess(DataNormalizer):
    def __init__(self, scaler_path=None, range_norm=(-0.5, 0.5), norm_pos=True, norm_vel=True, norm_acc=True):
        # Gọi hàm khởi tạo của class cha
        super().__init__(scaler_path, range_norm)
        self.norm_pos = norm_pos
        self.norm_vel = norm_vel
        self.norm_acc = norm_acc

    def vel_interpolation(self, x_arr, t_arr, method='spline-k3-s0'):
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

        if 'spline' in method:
            # tách method thành spline và bậc của spline từ string (check xem có đúng định dạng không: spline_number)
            s_value, k_value = self.get_s_k_values_from_string(method)
            # print(f"Found SPLINE: s: {s_value}, k: {k_value}")
            raw_f = UnivariateSpline(t_arr, x_arr, k=k_value, s=s_value)
            vel = raw_f.derivative(n=1)(t_arr)
            acc = raw_f.derivative(n=2)(t_arr)
                
        elif method == 'linear':
            vel = np.gradient(x_arr, t_arr)
        elif method == 'manual':
            vel = self.vel_interpolation_manual(x_arr, t_arr)
        else:
            raise ValueError("Invalid velocity interpolation method type")
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

    def acc_interpolation(self, x_arr, t_arr, vel_arr=None, method='spline-k3-s0'):
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
            s_value, k_value = self.get_s_k_values_from_string(method)
            # print(f"Found SPLINE: s: {s_value}, k: {k_value}")
            raw_f = UnivariateSpline(t_arr, x_arr, k=k_value, s=s_value)
            vel = raw_f.derivative(n=1)(t_arr)
            acc = raw_f.derivative(n=2)(t_arr)

        elif method == 'linear':
            # acc = np.gradient(vel_arr, t_arr)
            # 2nd derivative of position based on np.gradient() of position
            vel = np.gradient(x_arr, t_arr)  # Tính vận tốc
            acc = np.gradient(vel, t_arr)   # Tính gia tốc
        elif method == 'manual':
            acc = self.acc_interpolation_manual(x_arr, t_arr, vel_arr)
        else:
            raise ValueError("Invalid acceleration interpolation method type")

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

    def get_s_k_values_from_string(self, method):
        match = re.search(r'k(?P<k>\d+)-s(?P<s>[\d.]+)', method)
        if not match:
            raise ValueError("The string does not match the expected format 'k<number>-s<number>'")
        # Extract k and s
        s_value = None if match.group('s') == 'None' else float(match.group('s'))
        k_value = int(match.group('k'))
        return s_value, k_value
    
    # Hàm áp dụng Butterworth filter
    def apply_butterworth_for_axis(self, data, cutoff, fs, order=4):
        nyquist = 0.5 * fs  # Tần số Nyquist
        normal_cutoff = cutoff / nyquist  # Tần số cắt chuẩn hóa
        b, a = butter(order, normal_cutoff, btype='low', analog=False)  # Tạo bộ lọc
        filtered_data = filtfilt(b, a, data)  # Lọc tín hiệu
        return filtered_data
    
    def save_info_txt(self, ):
        pass

    def timestamp_interpolation(self, x, fs):
        t = []
        for i in range(x.shape[0]):
            t.append(i/fs)
        t = np.array(t)
        return t

    def run(self, trajectories, fs, cutoff):
        pass

    def save_processed_data(self, data, data_type, object_name):
        # get current path
        current_path = os.path.dirname(os.path.realpath(__file__))
        # get parent path
        parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
        output_dir = os.path.join(parent_path, 'data_preprocessed', object_name, f'{data_type}-{len(data)}')
        # create folder if not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for traj in data:
            idx = traj['original']['idx_org']
            # Save as .npz instead of .npy
            output_file = os.path.join(output_dir, f'pp_{object_name}_traj_{idx}.npz')
            # Save the whole trajectory dictionary in .npz format
            np.savez(output_file, **traj)
        global_util_printer.print_green(f'Saved processed data to {output_dir}')


    def apply_butterworth_filter(self, data, cutoff, freq_samp, butterworth_loop=1, debug=False):
        global_util_printer.print_blue('\n- Applying Butterworth filter', background=True)
        data_pp = []
        for idx, traj in enumerate(data):
            # traj Keys: frame_num, time_step, position, quaternion

            pos_seq_raw = traj['original']['position']
            
            # ---------------------------
            # 1. Apply Butterworth filter
            # ---------------------------

            
            # old_pos_seq = pos_seq_raw.copy()  # just for checking
            # new_pos_seq = pos_seq_raw.copy()
            old_pos_seq = copy.deepcopy(pos_seq_raw)    # use deepcopy to avoid changing the original data
            new_pos_seq = copy.deepcopy(pos_seq_raw)    # use deepcopy to avoid changing the original data
            # for x, y, z
            for i in range(3):
                for bl in range(butterworth_loop):
                    new_pos_seq[:, i] = self.apply_butterworth_for_axis(pos_seq_raw[:, i], cutoff, freq_samp, order=4)
            
            # check difference
            diff = np.where(old_pos_seq != new_pos_seq)  # just for checking
            if len(diff[0]) > 0:
                diff_steps = np.unique(diff[0])
                if debug: global_util_printer.print_yellow(f'There are changes after applying Butterworth filter: {len(diff_steps)}/{old_pos_seq.shape[0]} steps')
                d = new_pos_seq - old_pos_seq
                # check if the difference is significant
                small_change = True
                for step in diff_steps:
                    if np.linalg.norm(d[step]) > 0.01:
                        global_util_printer.print_red(f'Step {step} has significant difference: {np.linalg.norm(d[step])}')
                        small_change = False
                    
                    if debug:
                        # get random step in diff_steps to check
                        random_step = np.random.choice(diff_steps)
                        print('Random step: ', random_step)
                        print('     Old pos: ', old_pos_seq[random_step])
                        print('     New pos: ', new_pos_seq[random_step])

                    break
                if small_change:
                    if debug: global_util_printer.print_green('SMALL CHANGES')
                else:
                    if debug: global_util_printer.print_yellow('BIG CHANGES')
                    global_util_plotter.plot_trajectory_dataset_plotly([old_pos_seq, new_pos_seq], title='big change when applying butterworth', rotate_data_whose_y_up=True)
                    input('Press Enter to continue...')

            new_traj_dict = defaultdict(dict)
            new_traj_dict['original'] = traj['original']
            new_traj_dict['preprocess']['time_step'] = traj['original']['time_step']
            new_traj_dict['preprocess']['position'] = new_pos_seq
            new_traj_dict['object_name'] = traj['object_name']
            data_pp.append(new_traj_dict)
        return data_pp

    def vel_acc_interpolation(self, data, interpolate_method='spline-k3-s0'):
        global_util_printer.print_blue('\n- Interpolating velocities and accelerations', background=True)

        # check if data has available fields
        if 'position'in data[0]['preprocess']:
            source_field = 'preprocess'
        elif 'position' in data[0]['original']:
            global_util_printer.print_yellow('Cannot find position data in "preprocess" field. Do you want to use position data in "original" field instead [y/n] ?')
            ans = input()
            if ans == 'y':
                source_field = 'original'
            else:
                raise ValueError("No position data in preprocess data.")
        else:
            raise ValueError("No position data in preprocess data.")
        
        data_pp = []
        for idx, traj in enumerate(data):
            # traj Keys of field 'original': frame_num, time_step, position, quaternion
            # pos_seq = traj[source_field]['position'].copy()
            pos_seq = copy.deepcopy(traj[source_field]['position']) # use deepcopy to avoid changing the original data
            time_seq = traj[source_field]['time_step']

            vel_seq = []
            acc_seq = []
            # for x, y, z
            for i in range(3):
                vel_i = self.vel_interpolation(pos_seq[:, i], time_seq, method=interpolate_method)
                acc_i = self.acc_interpolation(pos_seq[:, i], time_seq, method=interpolate_method)
                vel_seq.append(vel_i)
                acc_seq.append(acc_i)
            vel_seq = np.array(vel_seq).T
            acc_seq = np.array(acc_seq).T
            # global_util_plotter.plot_line_chart(x_values=time_seq, y_values=[acc_seq[:, 0], acc_seq[:, 1], acc_seq[:, 2]], title='Acceleration - With ButterWorth filter', legends=['acc_x', 'acc_y', 'acc_z'])

            new_traj_dict = defaultdict(dict)
            new_traj_dict['original'] = traj['original']
            new_traj_dict['preprocess']['time_step'] = traj['original']['time_step']
            new_traj_dict['preprocess']['model_data'] = np.concatenate([pos_seq, vel_seq, acc_seq], axis=1)   # flatten data, each data point is a row with 9 features (3 positions, 3 velocities, 3 accelerations)
            new_traj_dict['object_name'] = traj['object_name']
            data_pp.append(new_traj_dict)
        return data_pp
    
    def detect_acc_outlier(self, data_raw, acc_threshold=(-20, 20), gap_threshold=3, edge_margin=25, min_len_threshold=65, plot_outlier=False, debug=False, check_temp=False):
        """
        Lọc ra và xử lý dữ liệu quỹ đạo xuất hiện nhiễu gia tốc:
        - Những quỹ đạo có nhiễu gia tốc ở phần đầu hoặc cuối sẽ được cắt bỏ.
        - Những quỹ đạo có nhiễu gia tốc ở giữa không thể xử lý được và được lưu vào biến outlier_trajectories

        Args:
            data_raw (list of dict): Dữ liệu quỹ đạo, mỗi phần tử là một dictionary chứa 'model_data'.
            acc_threshold (float): Ngưỡng giá trị gia tốc để xác định nhiễu.
            gap_threshold (int): Khoảng cách tối đa giữa hai điểm nhiễu để coi là một chuỗi nhiễu.
            edge_margin (int): Ngưỡng để kiểm tra xem một group nhiễu có nằm gần biên (đầu/cuối) của quỹ đạo hay không.
            min_len_threshold (int): Độ dài tối thiểu của quỹ đạo sau khi loại bỏ nhiễu, nếu nhỏ hơn sẽ bị loại bỏ.

        Returns:
            cleaned_data (list of dict): Dữ liệu đã được xử lý, loại bỏ các điểm nhiễu ở đầu và cuối.
            outlier_trajectories (default dict): Trajectory indices and their noisy indices, which cannot be cleaned and need to manually check.
        """
        global_util_printer.print_blue('\n- Detecting acceleration outliers', background=True)

        cleaned_data = []
        outlier_trajectories = defaultdict(list)
        traj_idxs_with_noise_treatment = []

        for traj_idx, traj in enumerate(data_raw):
            # global_util_printer.print_yellow(f'Processing traj {traj_idx}...')
            acc_data = traj['preprocess']['model_data'][:, 6:]  # Lấy dữ liệu gia tốc (ax, ay, az)
            len_traj_org = len(traj['preprocess']['model_data'])
            
            # 1. Tìm các chỉ số nhiễu
            noisy_indices = [
                i for i, acc in enumerate(acc_data)
                if any(a < acc_threshold[0] or a > acc_threshold[1] for a in acc)
            ]

            # print('noisy_indices len: ', len(noisy_indices))

            # 2. Lọc nhanh trước khi grouping
            if not noisy_indices:
                # Không có nhiễu, thêm quỹ đạo vào danh sách đã xử lý
                cleaned_data.append(traj)
                continue
            if len_traj_org - len(noisy_indices) < min_len_threshold:
                outlier_trajectories[traj_idx] = noisy_indices
                continue

            # 3. Grouping noisy indices
            ns_groups = self.group_noisy_indices(noisy_indices, gap_threshold)

            # 4. Noisy treatment
            '''
            We treat noisy data as follows:
            - If there exists noisy group in the middle of trajectory, we cannot clean it -> add to outlier_trajectories, continue to next trajectory.
            - If there exists noisy group at the beginning or end of trajectory, we remove it.
            '''
            # 4.1 Check if there exists noisy group in the middle of trajectory:
            # check if edge margin is available
            if edge_margin*2 > len_traj_org:
                raise ValueError("The edge_margin is too long to check noisy groups.")
            # classify by checking first and last index of each noisy group
            flag_middle_noise = False
            start_cut_idx = 0
            end_cut_idx = len_traj_org
            for gr in ns_groups:
                first_idx = gr[0]
                last_idx = gr[-1]

                # Check middle noise
                if  (edge_margin <= first_idx <= len_traj_org - edge_margin) or \
                    (edge_margin <= last_idx  <= len_traj_org - edge_margin):
                    flag_middle_noise = True
                    break
                # Check beginning noise
                if last_idx < edge_margin:
                    start_cut_idx = max(start_cut_idx, last_idx + 1)
                # Check end noise
                if first_idx > len_traj_org - edge_margin:
                    end_cut_idx = min(end_cut_idx, first_idx)

            
            if flag_middle_noise:
                outlier_trajectories[traj_idx] = noisy_indices
                continue
            # cut noisy group at the beginning or end of traj
            if start_cut_idx > 0 or end_cut_idx < len_traj_org:
                cleaned_traj = copy.deepcopy(traj)       
                cleaned_traj['preprocess']['model_data'] = traj['preprocess']['model_data'][start_cut_idx:end_cut_idx]
                cleaned_traj['preprocess']['time_step'] = traj['preprocess']['time_step'][start_cut_idx:end_cut_idx]

                # print('check cleaned_traj[preprocess] keys: ', cleaned_traj['preprocess'].keys())
                cleaned_data.append(cleaned_traj)
                traj_idxs_with_noise_treatment.append(traj_idx)
                if check_temp:
                    self.plot_acc(traj, note=f'before cut - Trajectory {traj_idx}')
                    self.plot_acc(cleaned_traj, note=f'after cut - Trajectory {traj_idx}')
                continue

        if debug:
            print('----------------- FILTER ACC OUTLIER RESULT -----------------')
            global_util_printer.print_yellow(f'Cleaned data count: {len(cleaned_data)}/{len(data_raw)}')
            global_util_printer.print_yellow(f'Bad trajectory count: {len(outlier_trajectories)}')
            # count number of trajectories whose None model_data
            global_util_printer.print_yellow(f'Number of treated trajectories: {len(traj_idxs_with_noise_treatment)}/{len(data_raw)} : {traj_idxs_with_noise_treatment}')
            for bad_idx, noisy_idxs in outlier_trajectories.items():
                global_util_printer.print_red('The following trajectories cannot be cleaned:')
                print(f'    Trajectory: {bad_idx}: {noisy_idxs} / {len(data_raw[bad_idx]["original"]["position"])}')
                if plot_outlier:
                    self.plot_traj(data_raw[bad_idx], title=f'Trajectory {bad_idx}', traj_type='raw')  # plot raw trajectory
                    # plot acceleration
                    self.plot_acc(data_raw[bad_idx], note=f'Need to manually check - Trajectory {bad_idx}')
            
            # set_none = input('Do you want to set None for bad trajectories? (y/n): ')
            # if set_none == 'y':
            #     for idx in outlier_trajectories.keys():
            #         data_pp[idx]['model_data'] = None
        
        return cleaned_data, outlier_trajectories, traj_idxs_with_noise_treatment

    def group_noisy_indices(self, noisy_indices, gap_threshold):
        """
        Nhóm các chỉ số nhiễu thành các nhóm dựa trên khoảng cách tối đa (gap_threshold).

        Args:
            noisy_indices (list[int]): Danh sách các chỉ số nhiễu đã được sắp xếp.
            gap_threshold (int): Khoảng cách tối đa giữa hai chỉ số để coi là một nhóm.

        Returns:
            list[list[int]]: Danh sách các nhóm chỉ số nhiễu.
        """
        if not noisy_indices:
            raise ValueError("The noisy_indices list needs to have at least one element to use this function.")
        noisy_indices = np.array(noisy_indices)
        gaps = np.diff(noisy_indices)  # Tính khoảng cách giữa các chỉ số
        group_boundaries = np.where(gaps >= gap_threshold)[0] + 1  # Tìm ranh giới giữa các nhóm
        return [list(group) for group in np.split(noisy_indices, group_boundaries)]
    
    def plot_traj(self, data, traj_type, title=''):
        '''
        traj_type: 'raw' or 'cleaned'
        '''
        if traj_type == 'raw':
            global_util_plotter.plot_trajectory_dataset_plotly(trajectories=[data['original']['position']], title=title, rotate_data_whose_y_up=True)
        elif traj_type == 'cleaned':
            global_util_plotter.plot_trajectory_dataset_plotly(trajectories=[data['model_data'][:, :3]], title=title, rotate_data_whose_y_up=True)
        else:
            raise ValueError("Invalid traj_type. Must be 'raw' or 'cleaned'.")

    def plot_acc(self, data, note=''):
        acc = data['preprocess']['model_data'][:, 6:]
        global_util_plotter.plot_line_chart(y_values=[acc[:, 0], acc[:, 1], acc[:, 2]], title=f'Acceleration - {note}', x_label='Frames', y_label='Acceleration m/s\u00b2', legends=['acc_x', 'acc_y', 'acc_z'])
    def plot_vel(self, data, note=''):
        vel = data['preprocess']['model_data'][:, 3:6]
        global_util_plotter.plot_line_chart(y_values=[vel[:, 0], vel[:, 1], vel[:, 2]], title=f'Velocity - {note}', x_label='Frames', y_label='Velocity m/s', legends=['vel_x', 'vel_y', 'vel_z'])
    def plot_pos(self, data, note=''):
        pos = data['preprocess']['model_data'][:, :3]
        global_util_plotter.plot_line_chart(y_values=[pos[:, 0], pos[:, 1], pos[:, 2]], title=f'Position - {note}', x_label='Frames', y_label='Position m', legends=['pos_x', 'pos_y', 'pos_z'])

    def backup_data(self, data, object_name):
        '''
        save original data to 'original' field
        '''
        for idx, traj in enumerate(data):
            traj_dict = defaultdict(dict)
            # traj_dict['original'] = {key: value.copy() for key, value in traj.items()}
            traj_dict['original'] = {key: copy.deepcopy(value) for key, value in traj.items()}  # use deepcopy to avoid changing original data
            traj_dict['original']['idx_org'] = idx
            traj_dict['object_name'] = object_name
            data[idx] = traj_dict
        global_util_printer.print_green(f'Backed up data to original field')
        return data

    def inspect_dict_structure(self, d, prefix=""):
        """
        Liệt kê tất cả các keys (kể cả keys con) trong một dictionary hoặc defaultdict.

        Args:
            d (dict or defaultdict): Dictionary cần kiểm tra.
            prefix (str): Chuỗi tiền tố để thể hiện key đầy đủ ở mỗi tầng.

        Returns:
            None
        """
        if not isinstance(d, dict):
            return  # Không làm gì nếu giá trị không phải dictionary

        for key in d.keys():
            print(f"  {prefix}{key}")  # In key hiện tại
            self.inspect_dict_structure(d[key], prefix=f"{prefix}{key}.")  # Gọi đệ quy cho keys con

    def normalize_data_features(self, data, object_name, debug=False):
        global_util_printer.print_blue('\n- Normalizing data (pos, vel, acc)', background=True)
        # Check data before normalization
        for idx, traj in enumerate(data):
            if 'model_data' not in traj['preprocess']:
                raise KeyError(f"'model_data' not found in trajectory {idx}")
            if traj['preprocess']['model_data'].shape[1] != 9:
                raise ValueError(f"Trajectory {idx} has fewer than 9 features: {traj['preprocess']['model_data'].shape}")

        # Remember the number of points in each trajectory
        len_list = [len(traj['preprocess']['model_data']) for traj in data]

        # Prepare to normalize position, velocity, and acceleration separately
        for norm_type, start_idx, end_idx, should_normalize in [
            ("position", 0, 3, self.norm_pos),
            ("velocity", 3, 6, self.norm_vel),
            ("acceleration", 6, 9, self.norm_acc),
        ]:
            if should_normalize:
                global_util_printer.print_blue(f"- Normalizing {norm_type}", background=True)
                # Extract the data to normalize
                feature_data_no_norm = [traj['preprocess']['model_data'][:, start_idx:end_idx].copy() for traj in data]
                feature_flatten = np.vstack(feature_data_no_norm)  # Flatten the data into a 2D array
                feature_flatten_normed_flatten = self.normalize_data(feature_flatten, object_name, note=norm_type)  # Normalize each column independently
                
                # Restore the data back into individual trajectories
                start = 0
                for idx, length in enumerate(len_list):
                    if data[idx]['preprocess']['model_data'][:, start_idx:end_idx].shape != feature_flatten_normed_flatten[start:start + length].shape:
                        raise ValueError(f"Mismatch in shapes for trajectory {idx} during {norm_type} normalization")
                    data[idx]['preprocess']['model_data'][:, start_idx:end_idx] = feature_flatten_normed_flatten[start:start + length]
                    start += length

        # Verify the correction after normalization
        for i in range(len(data)):
            if len(data[i]['preprocess']['model_data']) != len_list[i]:
                raise ValueError(f"Mismatch in shapes for trajectory {i} after normalization")
        
        # Debugging: Plot data before and after normalization
        if debug:
            idx_rand = 0  # Choose a random trajectory for visualization
            traj_rand = data[idx_rand]
            
            for norm_type, start_idx, end_idx, should_normalize in [
                ("position", 0, 3, self.norm_pos),
                ("velocity", 3, 6, self.norm_vel),
                ("acceleration", 6, 9, self.norm_acc),
            ]:
                if should_normalize:
                    # Before normalization
                    feature_no_norm = traj_rand['preprocess']['model_data'][:, start_idx:end_idx].copy()
                    global_util_plotter.plot_line_chart(
                        y_values=[feature_no_norm[:, i] for i in range(feature_no_norm.shape[1])],
                        title=f'{norm_type.capitalize()} - Before normalization - trajectory {idx_rand}',
                        legends=[f'{norm_type}_dim_{i}' for i in range(feature_no_norm.shape[1])]
                    )
                    # After normalization
                    feature_norm = traj_rand['preprocess']['model_data'][:, start_idx:end_idx]
                    global_util_plotter.plot_line_chart(
                        y_values=[feature_norm[:, i] for i in range(feature_norm.shape[1])],
                        title=f'{norm_type.capitalize()} - After normalization - trajectory {idx_rand}',
                        legends=[f'{norm_type}_dim_{i}' for i in range(feature_norm.shape[1])]
                    )

        return data
    
    def split_data(self, data, split_ratio_train_val_test=(0.8, 0.1, 0.1), shuffle=True):
        global_util_printer.print_blue('\n- Splitting data into train, val, test', background=True)
        # check ratio sum
        if sum(split_ratio_train_val_test) != 1:
            raise ValueError("The sum of split_ratio_train_val_test must be 1")
        if shuffle:
            for _ in range(5):
                random.shuffle(data)
        num_data = len(data)
        num_train = int(num_data * split_ratio_train_val_test[0])
        num_val = int(num_data * split_ratio_train_val_test[1])
        num_test = num_data - num_train - num_val
        data_train = data[:num_train]
        data_val = data[num_train:num_train + num_val]
        data_test = data[num_train + num_val:]
        global_util_printer.print_green(f'Split data into train: {num_train}, val: {num_val}, test: {num_test}')
        return data_train, data_val, data_test


def main():
    # data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/data/nae_paper_dataset/origin/trimmed_Bamboo_168'
    # object_name = 'Bamboo'

    data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/data/nae_paper_dataset/origin/trimmed_Bottle_115'
    object_name = 'Bottle'

    data_raw = RoCatDataRawReader(data_dir).read()
    FS = 120 # Sampling frequency
    CUTOFF = 25 # Cutoff frequency
    CUBIC_SPLINE = 'spline-k3-s0'
    RANGE_NORM = (-1, 1)
    NORM_POS = True
    NORM_VEL = True
    NORM_ACC = True
    data_preprocess = DataPreprocess(range_norm=RANGE_NORM, norm_pos=NORM_POS, norm_vel=NORM_VEL, norm_acc=NORM_ACC)
    s_value, k_value = data_preprocess.get_s_k_values_from_string(CUBIC_SPLINE)
    print(f"Found SPLINE configure: s: {s_value}, k: {k_value}")

    # ------------------------------------------------------------
    # 0. Create a new field 'original' in data_raw to store the original data
    # ------------------------------------------------------------
    data_pp = data_preprocess.backup_data(data_raw, object_name)

    # ------------------------------------------------------------
    # 1. Apply Butterworth filter and interpolation
    # ------------------------------------------------------------
    data_pp = data_preprocess.apply_butterworth_filter(data_pp, cutoff=CUTOFF, freq_samp=FS, butterworth_loop=1, debug=False)
    input('Done applying ButterWorth filter. Press Enter to continue...')

    # -------------------------------------------
    # 2. Interpolate velocities and accelerations
    # -------------------------------------------
    data_pp = data_preprocess.vel_acc_interpolation(data_pp, interpolate_method=CUBIC_SPLINE)
    input('Done interpolating velocities and accelerations. Press Enter to continue...')

    # ------------------------------------------------------------
    # 3. Filter out outlier trajectories with outlier acceleration
    # ------------------------------------------------------------
    cleaned_data, outlier_trajectories, traj_idxs_with_noise_treatment = data_preprocess.detect_acc_outlier(data_pp, acc_threshold=(-30, 30), gap_threshold=3, edge_margin=25, min_len_threshold=80, plot_outlier=True, debug=True)
    global_util_printer.print_yellow(f'Number of outlier trajectories: {len(outlier_trajectories)}')
    data_pp = cleaned_data
    input('Done filtering outlier trajectories. Press Enter to continue...')
                    
    acc_data_no_norm = data_pp[0]['preprocess']['model_data'][:, 6:].copy()
    # ----------------------------------------
    # 4. Normalize acceleration on all dataset
    # ----------------------------------------
    data_pp = data_preprocess.normalize_data_features(data_pp, object_name, debug=False)
    input('Done normalizing acceleration. Press Enter to continue...')

    # # plot acc
    # data_preprocess.plot_acc(data_pp[0], note='Trajectory 0 - After normalization'); input()
    # # plot vel
    # data_preprocess.plot_vel(data_pp[0], note='Trajectory 0 - After normalization'); input()
    # # plot pos
    # data_preprocess.plot_pos(data_pp[0], note='Trajectory 0 - After normalization'); input()

    # input('ENTER to continue...')
    # # plot acc
    # data_preprocess.plot_acc(data_pp[0], note='Trajectory 0 - After normalization'); input()

    # # check ratio of data after normalization
    # acc_data_norm = data_pp[0]['preprocess']['model_data'][:, 6:].copy()
    # for idx, acc_no_norm, acc_norm in zip(range(len(acc_data_no_norm)), acc_data_no_norm, acc_data_norm):
    #     print(f'{idx} - Ratio: {acc_no_norm[0] / acc_norm[0]} - {acc_no_norm[1] / acc_norm[1]} - {acc_no_norm[2] / acc_norm[2]}')
    #     input()

    # ----------------------------------------
    # 5. Split data into train, val, test
    # ----------------------------------------
    data_train, data_val, data_test = data_preprocess.split_data(data_pp, split_ratio_train_val_test=(0.8, 0.1, 0.1), shuffle=True)
    
    # -------------------------
    # 6. Save processed data
    # -------------------------
    # save processed data
    data_preprocess.save_processed_data(data=data_train, data_type='data_train', object_name=object_name)
    data_preprocess.save_processed_data(data=data_val, data_type='data_val', object_name=object_name)
    data_preprocess.save_processed_data(data=data_test, data_type='data_test', object_name=object_name)

if __name__ == '__main__':
    main()