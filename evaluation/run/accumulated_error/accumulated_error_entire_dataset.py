from nae_core.evaluation.NAE_metrics.accumulated_error_metric import *
from collections import defaultdict

'''
We will examine how the accumulated error changes with increasing input length
The input_seqs length is increased by 1 data point each time
The input_seqs includes input seqs with increasing length
(We will get mean accumulated error for each input length)
'''

def accumulated_err_filter(accumulated_err, filter_key='input_len', filter_step=1, range_filter=None):
    """
    Lọc dữ liệu accumulated_err theo filter_key và tính trung bình và độ lệch chuẩn.

    Args:
        accumulated_err (list): Danh sách các dict chứa 'filter_key' và 'accumulated_error'.
        filter_key (str): Khóa dùng để lọc ('input_len' hoặc 'len_left').
        filter_step (int): Bước lọc dữ liệu. Nếu = 1 -> không lọc
        range_filter (tuple): Bộ (min, max) để giới hạn khoảng giá trị của filter_key.

    Returns:
        list: Danh sách dict chứa filter_key, giá trị trung bình và độ lệch chuẩn của lỗi tích lũy.
    """

    min_val, max_val = range_filter
    # Tạo dictionary để nhóm dữ liệu
    grouped_data = defaultdict(list)
    for item in accumulated_err:
        key = item[filter_key]
        if filter_step <= 0:
            raise ValueError('filter_step must be greater than 0')

        if key % filter_step == 0 and key >= min_val and key <= max_val:
            grouped_data[key].append(item["accumulated_error"])

    # Tính giá trị trung bình và độ lệch chuẩn
    result = [
        {
            filter_key: key,
            "accumulated_error": np.mean(errors),
            "std": np.std(errors)
        }
        for key, errors in grouped_data.items()
    ]

    # Sắp xếp kết quả
    if filter_key == 'len_left':
        result.sort(key=lambda x: x[filter_key], reverse=True)  # Sắp xếp giảm dần
    elif filter_key == 'input_len':
        result.sort(key=lambda x: x[filter_key])  # Sắp xếp tăng dần

    return result



def main():
    metric = MetricAccumulatedError()
    device = torch.device('cuda')
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # --- Data and model ---
    id_traj = 'last 70 frames'
    filter_key = 'len_left'

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
    accumulated_err = metric.accumulated_error_cal(input_data, label_seqs, predicted_seqs)
    if accumulated_err == None:
        metric.util_printer.print_red(f'Error in accumulated error calculation', background=True)
        return
    
    accumulated_err_filtered = accumulated_err_filter(accumulated_err, filter_key=filter_key, filter_step=10, range_filter=(0, 70))
    # Print accumulated error
    plot = input('Do you want to check next trajectory [y/n] ? ')
    # 2. Plot
    if plot=='y':
        # 2.1 Show line chart of change in accumulated error with increasing input length
        input_lengths = [acer[filter_key] for acer in accumulated_err_filtered]
        mean_acc_errs = [acer['accumulated_error'] for acer in accumulated_err_filtered]
        acc_stds = [acer['std'] for acer in accumulated_err_filtered]
        if filter_key == 'len_left':
            label_x = 'Time to the goal (frame)'
            label_y = 'Prediction Error (Accumulated error) (m)'
        elif filter_key == 'input_len':
            label_x = 'Input length (data points)'
            label_y = 'Prediction Error (Accumulated error) (m)'
        
        # metric.util_plotter.plot_line_chart(x_values = input_lengths, y_values = [mean_acc_errs], y_stds=[acc_stds], 
        #                                 x_tick_distance=5, 
        #                                 y_tick_distance=0.01,
        #                                 font_size_title=32,
        #                                 font_size_label=24,
        #                                 font_size_tick=20,
        #                                 title=f'{thrown_object} - Accumulated error by input length - Trajectory #{id_traj}', 
        #                                 x_label=label_x, 
        #                                 y_label=label_y,
        #                                 legends=None,
        #                                 save_plot=True,
        #                                 keep_source_order=True,
        #                                 std_display_mode='bar')

        metric.util_plotter.plot_bar_chart(x_values = input_lengths, y_values = [mean_acc_errs], y_stds=[acc_stds], 
                                        x_tick_distance=5, 
                                        y_tick_distance=0.01,
                                        font_size_title=32,
                                        font_size_label=24,
                                        font_size_tick=20,
                                        title=f'{thrown_object} - Accumulated error by input length - Trajectory #{id_traj}', 
                                        x_label=label_x, 
                                        y_label=label_y,
                                        legends=None,
                                        save_plot=True,
                                        keep_source_order=True,
                                        bar_width=0.3)
    

if __name__ == '__main__':
    main()