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

from nae_core.utils.submodules.preprocess_utils.data_raw_reader import RoCatRLLabDataRawReader, RoCatNAEDataRawReader

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

    def normalize_data(self, data):
        if self.scaler is None:
            raise ValueError("Scaler chưa được khởi tạo.")
        data_normed = self.scaler.fit_transform(data)
        save_scaler = input('Do you want to save the scaler? (y/n): ')
        if save_scaler == 'y':
            self.save_scaler()
        return data_normed
    
    def denormalize_data(self, data_normalized):
        if self.scaler is None:
            raise ValueError("Scaler chưa được khởi tạo.")
        return self.scaler.inverse_transform(data_normalized)

    def save_scaler(self,):
        # get current path
        current_path = os.path.dirname(os.path.realpath(__file__))
        # get parent path
        parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
        parent_path = os.path.join(parent_path, 'data_preprocessed', 'normalization')
        # create folder if not exist
        if not os.path.exists(parent_path):
            os.makedirs(parent_path)
        # save new data
        scaler_path = os.path.join(parent_path, 'scaler.save')
        if self.scaler is None:
            raise ValueError("Scaler chưa được khởi tạo.")
        joblib.dump(self.scaler, scaler_path)
        global_util_printer.print_green(f'Saved scaler to {scaler_path}')
        input('Press Enter to continue...')

    def load_scaler(self, path):
        self.scaler = joblib.load(path)

class DataPreprocess(DataNormalizer):
    def __init__(self, scaler_path=None, range_norm=(-0.5, 0.5)):
        # Gọi hàm khởi tạo của class cha
        super().__init__(scaler_path, range_norm)

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
            match = re.search(r'spline-k(?P<k>\d+)-s(?P<s>\w+)', method)
            if not match:
                raise ValueError("The string does not contain 'univariate'")
            
            s_value = None if match.group('s') == 'None' else float(match.group('s'))
            k_value = int(match.group('k'))
            print(f"Found SPLINE: s: {s_value}, k: {k_value}")
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
            match = re.search(r'spline-k(?P<k>\d+)-s(?P<s>\w+)', method)
            if not match:
                raise ValueError("The string does not contain 'univariate'")
            
            s_value = None if match.group('s') == 'None' else float(match.group('s'))
            k_value = int(match.group('k'))
            print(f"Found SPLINE: s: {s_value}, k: {k_value}")
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

    def detect_acc_outlier(self, data_pp, acc_threshold=(-20, 20), gap_threshold=3, edge_margin=25, min_len_threshold=65, plot_outlier=False):
        """
        Lọc ra và xử lý dữ liệu quỹ đạo xuất hiện nhiễu gia tốc:
        - Những quỹ đạo có nhiễu gia tốc ở phần đầu hoặc cuối sẽ được cắt bỏ.
        - Những quỹ đạo có nhiễu gia tốc ở giữa không thể xử lý được và được lưu vào biến bad_trajectories

        Args:
            data_pp (list of dict): Dữ liệu quỹ đạo, mỗi phần tử là một dictionary chứa 'model_data'.
            acc_threshold (float): Ngưỡng giá trị gia tốc để xác định nhiễu.
            gap_threshold (int): Khoảng cách tối đa giữa hai điểm nhiễu để coi là một chuỗi nhiễu.
            edge_margin (int): Ngưỡng để kiểm tra xem một group nhiễu có nằm gần biên (đầu/cuối) của quỹ đạo hay không.
            min_len_threshold (int): Độ dài tối thiểu của quỹ đạo sau khi loại bỏ nhiễu, nếu nhỏ hơn sẽ bị loại bỏ.

        Returns:
            cleaned_data (list of dict): Dữ liệu đã được xử lý, loại bỏ các điểm nhiễu ở đầu và cuối.
            bad_trajectories (list of dict): List các index của các quỹ đạo có nhiễu ở giữa để xử lý riêng.
        """
        cleaned_data = []
        bad_trajectories = defaultdict(list)
        changed_idxs = []

        for traj_idx, trajectory in enumerate(data_pp):
            acc_data = trajectory['model_data'][:, 6:]  # Lấy dữ liệu gia tốc (ax, ay, az)
            len_traj_org = len(trajectory['model_data'])
            
            # 1. Tìm các chỉ số nhiễu
            noisy_indices = [
                i for i, acc in enumerate(acc_data)
                if any(a < acc_threshold[0] or a > acc_threshold[1] for a in acc)
            ]

            # 2. Lọc nhanh trước khi grouping
            if not noisy_indices:
                # Không có nhiễu, thêm quỹ đạo vào danh sách đã xử lý
                cleaned_data.append(trajectory)
                continue
            if len_traj_org - len(noisy_indices) < min_len_threshold:
                bad_trajectories[traj_idx] = noisy_indices
                changed_idxs.append(traj_idx)
                # trajectory['model_data'] = None
                continue

            # else: need to process noisy data
            # 3. Grouping noisy indices
            groups = []
            current_group = [noisy_indices[0]]
            # Nhóm các chỉ số nhiễu theo gap_threshold
            for idx in noisy_indices[1:]:
                if idx - current_group[-1] <= gap_threshold:
                    current_group.append(idx)
                else:
                    groups.append(current_group)
                    current_group = [idx]
            groups.append(current_group)

            # 4. Remove noise
            if len(groups) > 2:
                bad_trajectories[traj_idx] = noisy_indices
                changed_idxs.append(traj_idx)
                # trajectory['model_data'] = None
                continue

            # Chỉ còn lại 1 hoặc 2 nhóm nhiễu
            group_in_middle = False
            for group in groups:
                # nhiễu có thể nằm ở đâu hoặc cuối hoặc giữa
                # Xác định đoạn nhiễu này thuộc khu vực nào
                if group[0] > len_traj_org - edge_margin:   # Group nhiễu cuối
                    # Cắt bỏ các đoạn nhiễu ở cuối
                    trajectory['model_data'] = trajectory['model_data'][:group[0]]
                elif group[-1] < edge_margin:                                 # Group nhiễu đầu
                    # Cắt bỏ các đoạn nhiễu ở đầu
                    trajectory['model_data'] = trajectory['model_data'][group[-1]:]
                else:                                               # Group nhiễu giữa                                     
                    group_in_middle = True
                    break
            if group_in_middle:
                bad_trajectories[traj_idx] = noisy_indices
                changed_idxs.append(traj_idx)
                # trajectory['model_data'] = None
                continue
            
            if  len(trajectory['model_data']) < min_len_threshold:
                bad_trajectories[traj_idx] = noisy_indices
                changed_idxs.append(traj_idx)
                # trajectory['model_data'] = None
                continue

            cleaned_data.append(trajectory)
            changed_idxs.append(traj_idx)

        print('----------------- FILTER ACC OUTLIER RESULT -----------------')
        global_util_printer.print_yellow(f'Cleaned data count: {len(cleaned_data)}/{len(data_pp)}')
        global_util_printer.print_yellow(f'Bad trajectory count: {len(bad_trajectories)}')
        # count number of trajectories whose None model_data
        none_count = 0
        for traj in data_pp:
            if traj['model_data'] is None:
                none_count += 1
        global_util_printer.print_yellow(f'Trajectories with None model_data: {none_count}')
        global_util_printer.print_yellow(f'Changed trajectory count: {len(changed_idxs)}')
        for bad_idx, noisy_idxs in bad_trajectories.items():
            global_util_printer.print_red('The following trajectories cannot be cleaned:')
            print(f'    Trajectory: {bad_idx}: {noisy_idxs} / {len(data_pp[bad_idx]["position"])}')
            if plot_outlier:
                global_util_plotter.plot_trajectory_dataset_plotly(trajectories=[data_pp[bad_idx]['position']], title=f'Bad trajectory #{bad_idx}', rotate_data_whose_y_up=True)
                # plot acceleration
                acc = data_pp[bad_idx]['model_data'][:, 6:]
                global_util_plotter.plot_line_chart(x_values=data_pp[bad_idx]['time_step'], y_values=[acc[:, 0], acc[:, 1], acc[:, 2]], title=f'Acceleration - Trajectory {bad_idx}', x_label='Time step', y_label='Acceleration', legends=['acc_x', 'acc_y', 'acc_z'])
        set_none = input('Do you want to set None for bad trajectories? (y/n): ')
        if set_none == 'y':
            for idx in bad_trajectories.keys():
                data_pp[idx]['model_data'] = None
        return cleaned_data, bad_trajectories, changed_idxs



    def save_processed_data(self, data_pp, object_name):
        # get current path
        current_path = os.path.dirname(os.path.realpath(__file__))
        # get parent path
        parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
        output_dir = os.path.join(parent_path, 'data_preprocessed')
        # create folder if not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for idx, traj in enumerate(data_pp):
            np.save(os.path.join(output_dir, f'pp_{object_name}_traj_{idx}.npy'), traj, allow_pickle=True)
        global_util_printer.print_green(f'Saved processed data to {output_dir}')


    def apply_butterworth_filter_and_interpolation(self, data, cutoff, freq_samp, butterworth_loop=1, interpolate_method='spline-k3-s0'):
        data_pp = []
        for idx, traj in enumerate(data):
            # traj Keys: frame_num, time_step, position, quaternion

            global_util_printer.print_green(f'Processing trajectory {idx}', background=True)
            pos_seq = traj['position'].copy()
            vel_seq = []
            acc_seq = []
            
            # ---------------------------
            # 1. Apply Butterworth filter
            # ---------------------------
            old_pos_seq = pos_seq.copy()  # just for checking
            # for x, y, z
            for i in range(3):
                for bl in range(butterworth_loop):
                    # print(f'-------- {bl} --------')
                    # copy the raw data for checking
                    # pos_seq_i_raw = pos_seq[:, i].copy()
                    pos_seq[:, i] = self.apply_butterworth_for_axis(pos_seq[:, i], cutoff, freq_samp, order=4)
                    # # check if the filtered data is the same as the raw data based on np.allclose()
                    # diff_allclose = np.allclose(pos_seq_i_raw, pos_seq[:, i], atol=1e-4, rtol=1e-4)
                    # if diff_allclose:
                    #     global_util_printer.print_green('       diff_allclose NO CHANGES')
                    # else:
                    #     global_util_printer.print_yellow('       diff_allclose CHANGES')
                    # diff = np.where(pos_seq_i_raw != pos_seq[:, i])
                    # if len(diff[0]) > 0:
                    #     global_util_printer.print_yellow(f'      where CHANGE: {len(diff[0])}/{pos_seq.shape[0]} steps')
                    # else:
                    #     global_util_printer.print_green('       where NO CHANGES')
                    # input('check 111')
            
            # check difference
            new_pos_seq = pos_seq.copy()  # just for checking
            diff = np.where(old_pos_seq != new_pos_seq)  # just for checking
            if len(diff[0]) > 0:
                diff_steps = np.unique(diff[0])
                global_util_printer.print_yellow(f'There are changes after applying Butterworth filter: {len(diff_steps)}/{old_pos_seq.shape[0]} steps')
                d = new_pos_seq - old_pos_seq
                # check if the difference is significant
                small_change = True
                for step in diff_steps:
                    if np.linalg.norm(d[step]) > 0.01:
                        global_util_printer.print_red(f'Step {step} has significant difference: {np.linalg.norm(d[step])}')
                        small_change = False
                    # get random step in diff_steps to check
                    random_step = np.random.choice(diff_steps)
                    print('Random step: ', random_step)
                    print('     Old pos: ', old_pos_seq[random_step])
                    print('     New pos: ', new_pos_seq[random_step])
                    break
                if small_change:
                    global_util_printer.print_green('SMALL CHANGES')
                else:
                    global_util_printer.print_yellow('BIG CHANGES')
                    global_util_plotter.plot_trajectory_dataset_plotly([old_pos_seq, new_pos_seq], title='', rotate_data_whose_y_up=True)
                    input('Press Enter to continue...')

            # -------------------------------------------
            # 2. Interpolate velocities and accelerations
            # -------------------------------------------
            # for x, y, z
            for i in range(3):
                vel_i = self.vel_interpolation(pos_seq[:, i], traj['time_step'], method=interpolate_method)
                acc_i = self.acc_interpolation(pos_seq[:, i], traj['time_step'], method=interpolate_method)
                vel_seq.append(vel_i)
                acc_seq.append(acc_i)
            vel_seq = np.array(vel_seq).T
            acc_seq = np.array(acc_seq).T
            # global_util_plotter.plot_line_chart(x_values=traj['time_step'], y_values=[acc_seq[:, 0], acc_seq[:, 1], acc_seq[:, 2]], title='Acceleration - With ButterWorth filter', legends=['acc_x', 'acc_y', 'acc_z'])

            # create dictionary for each trajectory
            new_traj = {
                'time_step': traj['time_step'],
                'position': pos_seq,
                'velocity': vel_seq,
                'acceleration': acc_seq,
                'model_data': np.concatenate([pos_seq, vel_seq, acc_seq], axis=1)   # flatten data, each data point is a row with 9 features (3 positions, 3 velocities, 3 accelerations)
            }
            data_pp.append(new_traj)
        return data_pp

def main():
    data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/data/nae_paper_dataset/origin/trimmed_Bamboo_168'
    object_name = 'Bamboo'
    data_raw = RoCatNAEDataRawReader(data_dir).read()
    FS = 120 # Sampling frequency
    CUTOFF = 25 # Cutoff frequency
    CUBIC_SPLINE = 'spline-k3-s0'
    data_preprocess = DataPreprocess()
    enable_data_noise_filter = True

    while(enable_data_noise_filter):
        # ------------------------------------------------------------
        # 1. Apply Butterworth filter and interpolation
        # ------------------------------------------------------------
        data_pp = data_preprocess.apply_butterworth_filter_and_interpolation(data_raw, cutoff=CUTOFF, freq_samp=FS, butterworth_loop=1, interpolate_method=CUBIC_SPLINE)

        # ------------------------------------------------------------
        # 2. Filter out outlier trajectories with outlier acceleration
        # ------------------------------------------------------------
        cleaned_data, bad_trajectories, changed_idxs = data_preprocess.detect_acc_outlier(data_pp, acc_threshold=(-30, 30), gap_threshold=3, edge_margin=25, min_len_threshold=65, plot_outlier=True)
        global_util_printer.print_yellow(f'CHECK 111: len(bad_trajectories): {len(bad_trajectories)}')
        input('Done filtering outlier trajectories. Press Enter to continue...')

        # if len(bad_trajectories) == 0:
        #     enable_data_noise_filter = False


    input('Press Enter to continue...')
    

    # find trajectory with outlier acceleration
    # save outlier data into defaultdict
    # Tìm các trajectory có outliers
    outlier_idx = []
    for idx, data in enumerate(data_pp):  # Duyệt qua từng trajectory
        for step, acc_i in enumerate(data['model_data'][:, 6:]):  # Lấy acceleration
            if any(acc > 50 for acc in acc_i):  # Kiểm tra nếu bất kỳ giá trị nào > 10
                outlier_idx.append((idx, step))

    print('Outlier data count: ', len(outlier_idx))

    # Lưu các outliers vào defaultdict
    outlier_data = defaultdict(list)
    for idx, step in outlier_idx:
        outlier_data[idx].append(step)

    global_util_printer.print_yellow(f'There are {len(outlier_data)}/{len(data_pp)} trajectories with outlier acceleration')
    input('Do you want to see the outlier data? Press Enter to continue...')
    for idx, steps in outlier_data.items():
        global_util_printer.print_yellow(f'Trajectory {idx}: {len(data_pp[idx]["model_data"])}')
        print(f'        {steps}')
        if idx == 44 or idx == 83:
            # plot the outlier position trajectory
            global_util_plotter.plot_trajectory_dataset_plotly(trajectories=[data_pp[idx]['position']], title=f'trajectory #{idx}', rotate_data_whose_y_up=True)
            # plot acceleration
            # create x-axis with arange, not data_pp[idx]['time_step']
            x_axis = np.arange(len(data_pp[idx]['model_data']))
            global_util_plotter.plot_line_chart(x_values=x_axis, y_values=[data_pp[idx]['model_data'][:, 6], data_pp[idx]['model_data'][:, 7], data_pp[idx]['model_data'][:, 8]], title=f'Acceleration - Trajectory {idx}', legends=['acc_x', 'acc_y', 'acc_z'])
            input('Press Enter to continue...')
    input('Done checking outlier acceleration. Press Enter to continue...')
    # # check outlier acceleration values
    # for idx, traj in enumerate(data_pp):
    #     acc_data = traj['model_data'][:, 6:]
    #     for step, acc_i in enumerate(acc_data):
    #         if acc_i[0] > 20 or acc_i[1] > 20 or acc_i[2] > 20:
    #             global_util_printer.print_yellow(f'Trajectory {idx} has outlier acceleration at step {step}')
    #             print('Acceleration: ', acc_i)
    #             print('Position: ', traj['position'][step])
    #             print('Velocity: ', traj['velocity'][step])
    #             input('Press Enter to continue...')

        # for i in range(3):
        #     acc_i = acc_data[:, i]
        #     acc_i_mean = np.mean(acc_i)
        #     acc_i_std = np.std(acc_i)
        #     acc_i_outliers = np.where(np.abs(acc_i - acc_i_mean) > 3 * acc_i_std)
        #     if len(acc_i_outliers[0]) > 0:
        #         global_util_printer.print_yellow(f'Trajectory {idx} has {len(acc_i_outliers[0])} outliers in acceleration {i}')
        #         print('Outliers: ', acc_i_outliers)
        #         print('Mean: ', acc_i_mean)
        #         print('Std: ', acc_i_std)
        #         print('Max: ', np.max(acc_i))
        #         print('Min: ', np.min(acc_i))
        #         # check if the outliers are significant
        #         small_outliers = True
        #         for step in acc_i_outliers[0]:
        #             if np.abs(acc_i[step] - acc_i_mean) > 0.1:
        #                 global_util_printer.print_red(f'Step {step} has significant outlier: {acc_i[step]}')
        #                 small_outliers = False
        #                 break
        #         if small_outliers:
        #             global_util_printer.print_green('SMALL OUTLIERS')
        #         else:
        #             global_util_printer.print_yellow('BIG OUTLIERS')
        #             global_util_plotter.plot_line_chart(x_values=traj['time_step'], y_values=[acc_data[:, 0], acc_data[:, 1], acc_data[:, 2]], title=f'Acceleration - Trajectory {idx}', legends=['acc_x', 'acc_y', 'acc_z'])
        #             input('Press Enter to continue...')





                    
    # ----------------------------------------
    # 3. Normalize acceleration on all dataset
    # ----------------------------------------
    # Check data before normalization
    for idx, traj in enumerate(data_pp):
        if 'model_data' not in traj:
            raise KeyError(f"'model_data' not found in trajectory {idx}")
        if traj['model_data'].shape[1] != 9:
            raise ValueError(f"Trajectory {idx} has fewer than 9 features: {traj['model_data'].shape}")

    # Ghi nhớ số điểm trong từng quỹ đạo
    len_list = [len(traj['model_data']) for traj in data_pp]
    acc_data = [traj['model_data'][:, 6:] for traj in data_pp]
    acc_flatten = np.vstack(acc_data)  # Gộp toàn bộ quỹ đạo thành 1 mảng 2D
    # Normalize each column (x, y, z) independently
    acc_flatten_normed_flatten = data_preprocess.normalize_data(acc_flatten)
    # Tách lại dữ liệu thành danh sách các quỹ đạo
    start_idx = 0
    for idx, length in enumerate(len_list):
        if data_pp[idx]['model_data'][:, 6:].shape != acc_flatten_normed_flatten[start_idx:start_idx + length].shape:
            raise ValueError(f"Mismatch in shapes for trajectory {idx}")
        data_pp[idx]['model_data'][:, 6:] = acc_flatten_normed_flatten[start_idx:start_idx + length]
        start_idx += length


    # Plot before and after normalization
    idx_rand = 0
    traj_ran = data_pp[idx_rand]
    acc_x = traj_ran['acceleration'][:, 0]
    acc_y = traj_ran['acceleration'][:, 1]
    acc_z = traj_ran['acceleration'][:, 2]
    # global_util_plotter.plot_line_chart(x_values=traj_ran['time_step'], y_values=[acc_x, acc_y, acc_z], title=f'Acceleration - Before normalization - trajectory {idx_rand}', legends=['acc_x', 'acc_y', 'acc_z'])
    acc_x_norm = traj_ran['model_data'][:, 6]
    acc_y_norm = traj_ran['model_data'][:, 7]
    acc_z_norm = traj_ran['model_data'][:, 8]
    # global_util_plotter.plot_line_chart(x_values=traj_ran['time_step'], y_values=[acc_x_norm, acc_y_norm, acc_z_norm], title=f'Acceleration - After normalization - trajectory {idx_rand}', legends=['acc_x', 'acc_y', 'acc_z'])
    
    # check ratio of random steps
    random_steps = np.random.choice(traj_ran['time_step'], 10)
    for step in random_steps:
        idx = np.where(traj_ran['time_step'] == step)[0][0]
        print('Step: ', step)
        acc_raw = traj_ran['acceleration'][idx]
        acc_normed = traj_ran['model_data'][idx, 6:]
        acc_normed_denormed = data_preprocess.denormalize_data([acc_normed])
        # check if they are the same
        if not np.allclose(acc_raw, acc_normed_denormed):
            global_util_printer.print_red('Inaccurate normalization')
        else:
            global_util_printer.print_green('Accurate normalization')

        print('     Raw acc: ', acc_raw)
        print('     Normed acc: ', acc_normed)
        print('     Denormed acc: ', acc_normed_denormed)
        print('     Ratio: ', acc_normed / acc_raw)

        input('Press Enter to continue...')

    input()
    # -------------------------
    # 4. Save processed data
    # -------------------------


    # save processed data
    data_preprocess.save_processed_data(data_pp, object_name)

        # global_util_plotter.plot_trajectory_dataset_plotly([old_traj, new_traj], title='', rotate_data_whose_y_up=True)
        # input('Press Enter to continue with the next trajectory...')
if __name__ == '__main__':
    main()