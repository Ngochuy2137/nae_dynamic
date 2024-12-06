from nae_core.evaluation.NAE_metrics.accumulated_error_metric import *

'''
We will examine how the accumulated error changes with increasing input length
The input_seqs length is increased by 1 data point each time
The input_seqs includes input seqs with increasing length
(We will get mean accumulated error for each input length)
'''

def main():
    metric = MetricAccumulatedError()
    device = torch.device('cuda')
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # --- Data ---
    data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_prediction_ws/src/nae/data/rllab_dataset_no_orientation/data_enrichment/big_plane/big_plane_enrich_for_training'
    thrown_object = 'big_plane'

    # data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_prediction_ws/src/nae/data/nae_paper_dataset/new_data_format/bamboo/split/bamboo'
    # thrown_object = 'bamboo'

    # --- Model ---
    saved_model_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/models/big_plane-dynamic-len_model/NAE_DYNAMIC-model_02-12-2024_19-10-47_hiddensize128/@epochs260_data31529_batchsize128_hiddensize128_timemin195-45_NAE_DYNAMIC'

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

    for id_traj, traj in enumerate(data_test_raw):
        '''
        Consider each trajectory
        The trajectory is split into many input-label pairs
        '''
        input_label_generator = InputLabelGenerator()

        # Pairing the trajectory into many input-label pairs
        data_test = input_label_generator.generate_input_label_dynamic_seqs([traj], step_start=5, step_end=-3, increment=1, shuffle=False)
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
        
        plot = input('Do you want to check next trajectory? y/n')
        # 2. Plot
        if plot=='y':
            # 2.1 Show line chart of change in accumulated error with increasing input length
            input_lengths = [acer['input_len'] for acer in accumulated_err]
            mean_acc_errs = [acer['accumulated_error'] for acer in accumulated_err]
            metric.util_plotter.plot_line_chart(x_values = input_lengths, y_values = [mean_acc_errs], 
                                            x_tick_distance=5, 
                                            y_tick_distance=0.02,
                                            font_size_title=32,
                                            font_size_label=24,
                                            font_size_tick=20,
                                            title=f'{thrown_object} - Accumulated error by input length - Trajectory #{id_traj}', 
                                            x_label='Input length (data points)', 
                                            y_label='Accumulated error (m)',
                                            legends=None,
                                            save_plot=True)
            # 2.2 Show the predictions in 3D plot
            metric.util_plotter.plot_predictions_plotly(inputs=input_data, labels=label_seqs, predictions=predicted_seqs, 
                                                        title=f'{thrown_object} - Predictions - Trajectory #{id_traj}', rotate_data_whose_y_up=True, 
                                                        save_plot=False, font_size_note=12,
                                                        show_all_as_default=False)
    
        
        input('Do you want to check next trajectory?')

if __name__ == '__main__':
    main()