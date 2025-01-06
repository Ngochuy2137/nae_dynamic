from nae_core.nae_dynamic import NAEDynamicLSTM
from nae_core.utils.submodules.training_utils.data_loader import DataLoader as NAEDataLoader
from nae_core.utils.submodules.training_utils.input_label_generator import InputLabelGenerator

from abc import ABC, abstractmethod
import numpy as np
from python_utils.printer import Printer
from python_utils.plotter import Plotter
from collections import defaultdict
import torch 
import random
import glob
import os
import json

class Metric(ABC):
    """
    Base class for evaluation metrics.
    """
    def __init__(self):
        self.util_printer = Printer()
        self.util_plotter = Plotter()

    def is_proper_prediction(self, input_data, label_seqs):
        # We determine based on the input data and the label data
        for in_check, la_check in zip(input_data, label_seqs):
            in_check_cut = np.array(in_check[1:])
            la_check_cut = np.array(la_check[:in_check_cut.shape[0]])
            same = np.allclose(in_check_cut, la_check_cut)
            if not same:
                self.util_printer.print_red(f'{len(input_data)} labels are incorrect', background=True)
                self.util_plotter.plot_predictions([in_check], [la_check], lim_plot_num=None, rotate_data_whose_y_up=False, title=f'{same}')
                return False
        return True
    
    # @abstractmethod
    # def compute(self):
    #     """
    #     Compute and return the metric value.
    #     """
    #     pass

    @abstractmethod
    def process_and_plot(self):
        """
        Process the computed metric and plot the result.
        """
        pass

    def post_process(self, result_raw, value_filter, key_filter='input_len', filter_step=1, range_filter=None):
        """
        Lọc dữ liệu result_raw theo key_filter và tính trung bình và độ lệch chuẩn.

        Args:
            result_raw (list): Danh sách các dict chứa 'key_filter' và 'error'.
            key_filter (str): Khóa dùng để lọc ('input_len' hoặc 'len_left').
            filter_step (int): Bước lọc dữ liệu. Nếu = 1 -> không lọc
            range_filter (tuple): Bộ (min, max) để giới hạn khoảng giá trị của key_filter.

        Returns:
            list: Danh sách dict chứa key_filter, giá trị trung bình và độ lệch chuẩn của lỗi tích lũy.
        """

        min_val, max_val = range_filter
        # Tạo dictionary để nhóm dữ liệu
        # Group all data with same key_filter to a list
        grouped_data = defaultdict(list)
        for item in result_raw:
            key = item[key_filter]
            if filter_step <= 0:
                raise ValueError('filter_step must be greater than 0')

            if key % filter_step == 0 and key >= min_val and key <= max_val:
                grouped_data[key].append(item[value_filter])

        # Tính giá trị trung bình và độ lệch chuẩn
        result = [
            {
                key_filter: key,
                value_filter: np.mean(errors),
                "std": np.std(errors)
            }
            for key, errors in grouped_data.items()
        ]

        # Sắp xếp kết quả
        if key_filter == 'len_left':
            result.sort(key=lambda x: x[key_filter], reverse=True)  # Sắp xếp giảm dần
        elif key_filter == 'input_len':
            result.sort(key=lambda x: x[key_filter])  # Sắp xếp tăng dần

        return result
    
    def compute_impact_point_err(self, pred, lab):
        # Only calculate the error for last point
        lab_last = lab[-1, :3]
        pred_last = pred[-1, :3]
        goal_err = np.linalg.norm(pred_last - lab_last)
        return goal_err

    def compute_accumulated_error(self, lab, pred):
        # Only calculate the accumulated error for the first 3 dimensions x, y, z
        lab = lab[:, :3]
        pred = pred[:, :3]
        dis_all = np.linalg.norm(pred - lab, axis=-1)
        accumulated_error = np.mean(dis_all)
        return accumulated_error
    
    def compute(self, input_seqs, label_seqs, predicted_seqs):
        # 0. check if the prediction is proper:
        if not self.is_proper_prediction(input_seqs, label_seqs):
            self.util_printer.print_red(f'{len(input_seqs)} labels are incorrect')
            return
        self.util_printer.print_green(f'{len(input_seqs)} labels are correct')

        # 1. Calculate accumulated error for each prediction
        err_by_in_len = []
        # count = 0

        # Consider one group (input, label, predicted) at a time
        for inp, pred, lab in zip(input_seqs, predicted_seqs, label_seqs):
            # Only calculate the error for last point
            last_err = self.compute_impact_point_err(pred, lab)
            accumulated_error = self.compute_accumulated_error(lab, pred)

            err_element = {
                'input_len': len(inp),
                'len_left': len(lab) - len(inp),
                'impact_point_err': last_err,
                'accumulated_error': accumulated_error
            }
            err_by_in_len.append(err_element) 
        return err_by_in_len
    
    def load_model_config(self, folder_path):
        """
        Tìm file và đọc file model_config.json, lưu nội dung vào dictionary.
        """
        try:
            # Tìm file model_config.json trong thư mục
            for root, _, files in os.walk(folder_path):
                if "model_config.json" in files:
                    config_path = os.path.join(root, "model_config.json")
                    # Đọc nội dung file model_config.json
                    with open(config_path, "r", encoding="utf-8") as f:
                        config_dict = json.load(f)
                    return config_dict

            # Nếu không tìm thấy file
            raise ValueError("Cannot find model_config.json")

        except Exception as e:
            raise ValueError(f"Something is wrong with error: {e}")


    def search_model_at_epoch(self, parent_dir, epoch_idx):
        saved_model_dir = glob.glob(f'{parent_dir}/*epochs{epoch_idx}_*')
        if len(saved_model_dir) == 0:
            self.util_printer.print_red(f'Cannot find the model with epoch {epoch_idx}', background=True)
            return None
        elif len(saved_model_dir) > 1:
            self.util_printer.print_red(f'More than one model with epoch {epoch_idx}', background=True)
            for i, model_dir in enumerate(saved_model_dir):
                self.util_printer.print_yellow(f'   {i+1}. {model_dir}')
            raise ValueError(f'More than one model with epoch {epoch_idx}')
        else:
            return saved_model_dir[0]

class MetricAccumulatedError(Metric):
    def __init__(self):
        super().__init__()  # call the parent class constructor
        
    
class MetricAccumulatedError(Metric):
    def __init__(self):
        super().__init__()  # call the parent class constructor
        
    def compute(self, input_seqs, label_seqs, predicted_seqs):
        # 0. check if the prediction is proper:
        if not self.is_proper_prediction(input_seqs, label_seqs):
            self.util_printer.print_red(f'{len(input_seqs)} labels are incorrect')
            return
        self.util_printer.print_green(f'{len(input_seqs)} labels are correct')

        # 1. Calculate accumulated error for each prediction
        impact_point_err_by_length = []
        # count = 0

        # Consider one group (input, label, predicted) at a time
        for inp, pred, lab in zip(input_seqs, predicted_seqs, label_seqs):
            # Only calculate the error for last point
            last_err = self.compute_impact_point_err(pred, lab)
            accumulated_error = self.compute_accumulated_error(lab, pred)

            err_by_inlen = {
                'input_len': len(inp),
                'len_left': len(lab) - len(inp),
                'impact_point_err': last_err,
                'accumulated_error': accumulated_error
            }
            impact_point_err_by_length.append(err_by_inlen) 
        return err_by_inlen

class MetricAccumulatedError(Metric):
    def __init__(self):
        super().__init__()  # call the parent class constructor
        
    def process_and_plot(self, input_seqs, label_seqs, predicted_seqs, id_traj, thrown_object, filter_value, epoch_idx, filter_key='len_left'):
        # convert all elements of input_seqs to numpy
        input_seqs = [inp.cpu().numpy() for inp in input_seqs]

        # 1. Calculate error
        metric_result = self.compute(input_seqs, label_seqs, predicted_seqs)
        if metric_result == None:
            self.util_printer.print_red(f'Error in metric calculation', background=True)
            return
        
        result_filtered = self.post_process(metric_result, key_filter=filter_key, value_filter=filter_value, filter_step=10, range_filter=(0, 70))
        plot = input('Do you want to plot trajectory [y/n] ? ')
        answer_plot = input('Do you want to save the plot [y/n] ? ')
        if answer_plot == 'y':
            save_plot = True
        else:
            save_plot = False

        # 2. Plot
        if plot=='y':
            # 2.1 Show line chart of change in error with increasing input length
            x_plot = [rf[filter_key] for rf in result_filtered]
            mean_errs = [rf[filter_value] for rf in result_filtered]
            acc_stds = [rf['std'] for rf in result_filtered]
            if filter_key == 'len_left':
                label_x = 'Time to the goal (frame)'
            elif filter_key == 'input_len':
                label_x = 'Input length (data points)'
            label_y = f'Prediction Error ({filter_value}) (m)'

            self.util_plotter.plot_bar_chart(x_values = x_plot, y_values = [mean_errs], y_stds=[acc_stds], 
                                            # x_tick_distance=5, 
                                            # y_tick_distance=0.01,
                                            font_size_title=32,
                                            font_size_label=20,
                                            font_size_tick=20,
                                            font_size_bar_val=22,
                                            title=f'{thrown_object} - {filter_value} by input length - #{id_traj} - EPOCH {epoch_idx}', 
                                            x_label=label_x, 
                                            y_label=label_y,
                                            legends=None,
                                            save_plot=save_plot,
                                            # keep_source_order=True,
                                            bar_width=0.3)
    
class MetricGoalError(Metric):
    def __init__(self):
        super().__init__()  # call the parent class constructor
        
    def process_and_plot(self, input_seqs, label_seqs, predicted_seqs, id_traj, thrown_object, filter_value, epoch_idx, filter_key='len_left', note=''):
        # convert all elements of input_seqs to numpy
        input_seqs = [inp.cpu().numpy() for inp in input_seqs]

        # 1. Calculate error
        metric_result = self.compute(input_seqs, label_seqs, predicted_seqs)
        if metric_result == None:
            self.util_printer.print_red(f'Error in metric calculation', background=True)
            return
        
        result_filtered = self.post_process(metric_result, key_filter=filter_key, value_filter=filter_value, filter_step=1, range_filter=(0, 80))
        plot = input('Do you want to plot trajectory [y/n] ? ')
        answer_plot = input('Do you want to save the plot [y/n] ? ')
        if answer_plot == 'y':
            save_plot = True
        else:
            save_plot = False

        # 2. Plot
        if plot=='y':
            # 2.1 Show line chart of change in error with increasing input length
            x_plot = [rf[filter_key] for rf in result_filtered]
            mean_errs = [rf[filter_value] for rf in result_filtered]
            acc_stds = [rf['std'] for rf in result_filtered]
            if filter_key == 'len_left':
                label_x = 'Time to the goal (frame)'
            elif filter_key == 'input_len':
                label_x = 'Input length (data points)'
            label_y = f'Prediction Error ({filter_value}) (m)'
            
            self.util_plotter.plot_line_chart(x_values = x_plot, y_values = [mean_errs], y_stds=[acc_stds], 
                                            x_tick_distance=5, 
                                            y_tick_distance=0.01,
                                            font_size_title=40,
                                            font_size_label=24,
                                            font_size_tick=20,
                                            title=f'{thrown_object} - Impact point error - Trajectory #{id_traj} - EPOCH {epoch_idx} - {note}', 
                                            x_label=label_x, 
                                            y_label=label_y,
                                            legends=None,
                                            save_plot=save_plot,
                                            keep_source_order=True,
                                            std_display_mode='fill')

class MetricLeadingtime(Metric):
    def __init__(self):
        pass

def main():
    metric = MetricAccumulatedError()
    device = torch.device('cuda')
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    id_traj = 'last 70 frames'
    filter_key = 'len_left'
    filter_value = 'accumulated_error'

    # --- Data and model ---
    # data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_prediction_ws/src/nae/data/rllab_dataset_no_orientation/data_enrichment/big_plane/big_plane_enrich_for_training'
    # thrown_object = 'big_plane'
    # saved_model_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/models/big_plane-dynamic-len_model/NAE_DYNAMIC-model_02-12-2024_19-10-47_hiddensize128/@epochs260_data31529_batchsize128_hiddensize128_timemin195-45_NAE_DYNAMIC'

    data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_prediction_ws/src/nae/data/nae_paper_dataset/new_data_format/bamboo/split/bamboo'
    thrown_object = 'bamboo'
    saved_model_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/models/bamboo-dynamic-len_model/NAE_DYNAMIC-model_05-12-2024_21-17-51_hiddensize128/@epochs480_data14770_batchsize256_hiddensize128_timemin76-37_NAE_DYNAMIC'

    # Training parameters 
    training_params = {
        'num_epochs': 5000,
        'batch_size_train': 128,    
        'batch_size_val': 1024,
        'save_interval': 10,
        'thrown_object' : thrown_object + '-dynamic-len'
    }
    # Model parameters
    model_params = {
        'input_size': 9,
        'hidden_size': 128,
        'output_size': 9,
        'num_layers_lstm': 2,
        'lr': 0.0001
    }

    nae = NAEDynamicLSTM(**model_params, **training_params, data_dir=data_dir, device=device)
    # load data
    nae_data_loader = NAEDataLoader()
    data_train_raw, data_val_raw, data_test_raw = nae_data_loader.load_dataset(data_dir)
    if not nae.data_correction_check(data_train_raw, data_val_raw, data_test_raw):
        return
    # prepare data for training
    nae.load_model(saved_model_dir, weights_only=True)

    '''
    Consider all trajectory in the test dataset
    The trajectory is split into many input-label pairs
    '''
    input_label_generator = InputLabelGenerator()

    # Pairing the trajectory into many input-label pairs
    data_test = input_label_generator.generate_input_label_dynamic_seqs(data_test_raw, step_start=5, step_end=-3, increment=1, shuffle=False)
    # Inference
    predicted_seqs, label_seqs = nae.validate_and_score(data=data_test, batch_size=1024, shuffle=False, inference=True)

    print(f'There {len(predicted_seqs)} groups of input-label-prediction sequences')

    input_data = [inp[0] for inp in data_test]
    # convert all elements of input_data to numpy
    input_data = [inp.cpu().numpy() for inp in input_data]

    # 1. Calculate accumulated error for one trajectory
    accumulated_err = metric.compute(input_data, label_seqs, predicted_seqs)
    if accumulated_err == None:
        metric.util_printer.print_red(f'Error in accumulated error calculation', background=True)
        return
    
    accumulated_err_filtered = metric.process_to_plot(accumulated_err, key_filter=filter_key, value_filter=filter_value, filter_step=10, range_filter=(0, 70))
    # Print accumulated error
    plot = input('Do you want to plot trajectory [y/n] ? ')
    save_plot = input('Do you want to save the plot [y/n] ? ')
    if save_plot == 'y':
        save_plot = True
    else:
        save_plot = False
    # 2. Plot
    if plot=='y':
        # 2.1 Show line chart of change in accumulated error with increasing input length
        x_plot = [acer[filter_key] for acer in accumulated_err_filtered]
        mean_acc_errs = [acer['accumulated_error'] for acer in accumulated_err_filtered]
        acc_stds = [acer['std'] for acer in accumulated_err_filtered]
        if filter_key == 'len_left':
            label_x = 'Time to the goal (frame)'
            label_y = 'Prediction Error (Accumulated error) (m)'
        elif filter_key == 'input_len':
            label_x = 'Input length (data points)'
            label_y = 'Prediction Error (Accumulated error) (m)'

        metric.util_plotter.plot_bar_chart(x_values = x_plot, y_values = [mean_acc_errs], y_stds=[acc_stds], 
                                        x_tick_distance=5, 
                                        y_tick_distance=0.01,
                                        font_size_title=32,
                                        font_size_label=24,
                                        font_size_tick=20,
                                        title=f'{thrown_object} - Accumulated error by input length - Trajectory #{id_traj}', 
                                        x_label=label_x, 
                                        y_label=label_y,
                                        legends=None,
                                        save_plot=save_plot,
                                        keep_source_order=True,
                                        bar_width=0.3)

if __name__ == '__main__':
    main()