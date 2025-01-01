from nae_core.evaluation.NAE_metrics.metric import *
import glob

'''
We will examine how the accumulated error changes with increasing input length
The input_seqs length is increased by 1 data point each time
The input_seqs includes input seqs with increasing length
(We will get mean accumulated error for each input length)
'''

def main():
    acc_metric = MetricAccumulatedError()
    ge_metric = MetricGoalError()
    device = torch.device('cuda')
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    ## ----------------- 1. Bamboo -----------------
    data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/data/nae_paper_dataset/data_preprocessed/Bamboo'
    thrown_object = 'bamboo'
    parent_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/models/ACC-repair-bamboo_model/bamboo-model_01-01-2025_06-58-11_hiddensize128'
    epoch_idx = 1220 # 1210 1220
    saved_model_dir = glob.glob(f'{parent_dir}/*epochs{epoch_idx}*')[0]


    # # ## ----------------- 3. Bottle -----------------\
    # data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/data/nae_paper_dataset/data_preprocessed/Bottle'
    # thrown_object = 'bottle'
    # parent_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/models/ACC-repair-bottle-lr5e-5_model/bottle-lr5e-5-model_01-01-2025_22-28-52_hiddensize128'
    # epoch_idx = 6500
    # saved_model_dir = glob.glob(f'{parent_dir}/*epochs{epoch_idx}*')[0]

    
    # Training parameters 
    training_params = {
        'num_epochs': 5000,
        'batch_size_train': 128,    
        'batch_size_val': 1024,
        'save_interval': 10,
        'thrown_object' : thrown_object + '-dynamic-len',
        'train_id': None,
    }
    # Model parameters
    model_params = {
        'input_size': 9,
        'hidden_size': 128,
        'output_size': 9,
        'num_layers_lstm': 2,
        'lr': 0.0001,
        'dropout_rate': 0.3
    }

    nae = NAEDynamicLSTM(**model_params, **training_params, data_dir=data_dir, device=device)
    # load data
    nae_data_loader = NAEDataLoader()
    data_train_raw, data_val_raw, data_test_raw = nae_data_loader.load_train_val_test_dataset(data_dir)
    # prepare data for training
    nae.load_model(saved_model_dir, weights_only=True)

    for id_traj, traj in enumerate(data_test_raw):
        # if id_traj != 5:
        #     continue
        '''
        Consider each trajectory
        The trajectory is split into many input-label pairs
        '''
        input_label_generator = InputLabelGenerator()

        # Pairing the trajectory into many input-label pairs
        data_test = input_label_generator.generate_input_label_dynamic_seqs([traj], step_start=5, step_end=-3, increment=1, shuffle=False)
        # Inference
        predicted_seqs, label_seqs, final_err_var_penalty = nae.validate_and_score(data=data_test, batch_size=1024, shuffle=False, inference=True)

        print('\n---------------------------------------------------')
        print('Considering trajectory: ', id_traj)
        print(f'    There {len(predicted_seqs)} groups of input-label-prediction sequences')

        input_data = [inp[0] for inp in data_test]
        # convert all elements of input_data to numpy
        input_data = [inp.cpu().numpy() for inp in input_data]

        

        # ---------------------------------------------------
        # 1. Calculate accumulated error and goal error
        # ---------------------------------------------------
        accumulated_err = acc_metric.compute(input_data, label_seqs, predicted_seqs)
        if accumulated_err == None:
            acc_metric.util_printer.print_red(f'Error in acc_metric calculation', background=True)
            return
        

        goal_err = ge_metric.compute(input_data, label_seqs, predicted_seqs)

        
        # plot = input('    Do you want to plot [y/n] ? ')
        plot='y'
        # 2. Plot
        if plot=='y':
            # 2.1 Show line chart of change in accumulated error with increasing input length
            input_lengths = [acer['input_len'] for acer in accumulated_err]
            mean_acc_errs = [acer['accumulated_error'] for acer in accumulated_err]
            acc_metric.util_plotter.plot_line_chart(x_values = input_lengths, y_values = [mean_acc_errs], 
                                            x_tick_distance=5, 
                                            y_tick_distance=0.02,
                                            font_size_title=32,
                                            font_size_label=24,
                                            font_size_tick=20,
                                            title=f'{thrown_object} - Accumulated error by input length - Trajectory #{id_traj}', 
                                            x_label='Input length (data points)', 
                                            y_label='Accumulated error (m)',
                                            legends=None,
                                            save_plot=False)
            # 2.2 Show the predictions in 3D plot
            acc_metric.util_plotter.plot_predictions_plotly(inputs=input_data, labels=label_seqs, predictions=predicted_seqs, 
                                                        title=f'{thrown_object} - Predictions - Trajectory #{id_traj}', rotate_data_whose_y_up=True, 
                                                        save_plot=False, font_size_note=12,
                                                        show_all_as_default=False)
    
        
        input('    Do you want to check next trajectory [y/n] ? ')

if __name__ == '__main__':
    main()