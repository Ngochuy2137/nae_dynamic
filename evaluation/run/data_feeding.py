from nae_core.evaluation.NAE_metrics.metric import *
import glob
from python_utils.printer import Printer
global_util_printer = Printer()

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

    # ## ----------------- 1. Bamboo -----------------
    # data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/data/nae_paper_dataset/data_preprocessed/Bamboo'
    # thrown_object = 'bamboo'
    # parent_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/models/ACC-repair-bamboo_model/bamboo-model_01-01-2025_06-58-11_hiddensize128'
    # epoch_idx = 1220 # 1210 1220
    # saved_model_dir = glob.glob(f'{parent_dir}/*epochs{epoch_idx}*')[0]


    # ## ----------------- 3. Bottle -----------------\
    data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/data/nae_paper_dataset/data_preprocessed/norm_acc/Bottle'
    thrown_object = 'bottle'
    # # ----- lr = 5e-5
    # parent_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/models/ACC-repair-bottle-lr5e-5_model/bottle-lr5e-5-model_01-01-2025_22-28-52_hiddensize128'
    # epoch_idx = 5530 #6900
    # note = 'lr: 5e-5'

    ## ----- lr = 1e-4
    # parent_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/models/ACC-repair-bottle-lr1e-4_model/bottle-lr1e-4-model_01-01-2025_23-58-41_hiddensize128'
    # epoch_idx = 4680    #   3600 4680 5360
    # note = 'lr: 1e-4'

    ## ----- 2.0 * Loss1
    # parent_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/models/ACC-repair-bottle-2loss1_model/bottle-2loss1-model_02-01-2025_09-15-20_hiddensize128'
    # epoch_idx = 4610    # 4350 5020 6990 7420 11310
    # note = '2.0 * Loss1'

    # ## ----- 3.0 * Loss1
    # parent_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/models/ACC-repair-bottle-3loss1_model/bottle-3loss1-model_02-01-2025_12-04-38_hiddensize128'
    # epoch_idx = 5000    # 4600
    # note = '3.0 * Loss1'


    # ## ====== inlen 25 !=5 ======
    # parent_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/models/bottle-1loss1-inlen-25_model/ACC-repair-bottle-1loss1-inlen-25-model_03-01-2025_02-16-45_hiddensize128'
    # epoch_idx = 4800 # 1080 4800
    # note = '1.0 * Loss1, inlen 25'


    # # ## ====== inlen 25, more layers, GRA, CLIP ======
    # parent_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/models/3-5-1-bottle-1loss1-inlen-25-wc1e-4-more-layers-hidden-GRA-CLIP_model/ACC-repair-3-5-1-bottle-1loss1-inlen-25-wc1e-4-more-layers-hidden-GRA-CLIP-model_03-01-2025_19-21-16_hiddensize256'
    # epoch_idx = 5700 # 5700 8880
    # note = '1.0 * Loss1, inlen 25, more layers, GRA, CLIP'

    ## ================= OLD CONFIG =================
    ## Training parameters 
    # training_params = {
    #     'num_epochs': 12000,
    #     'batch_size_train': 512,    
    #     'save_interval': 10,
    #     'thrown_object' : thrown_object,
    #     'train_id': 'ACC-repair',
    #     'warmup_steps': 25,
    #     'dropout_rate': 0.0,
    #     'loss1_weight': 1.0,
    #     'loss2_weight': 1.0,
    #     'loss2_1_weight': 0.0,
    #     'weight_decay': 0.0001,
    # }
    # ## Model parameters
    # model_params = {
    #     'input_size': 9,
    #     'hidden_size': 128,
    #     'output_size': 9,
    #     'num_layers_lstm': 2,
    #     'lr': 0.0001
    # }

    # data_params = {
    #     'data_step_start': 1,
    #     'data_step_end': -1,
    #     'data_increment': 1,
    # }
    # nae = NAEDynamicLSTM(**model_params, **training_params, **data_params, data_dir=data_dir, device=device)
    # saved_model_dir = metric.search_model_at_epoch(parent_dir, epoch_idx)
    # step_start=5
    # step_end=-3
    # increment=1



    # ================ NEW CONFIG =================
    # ## 3-5-1-2
    # parent_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/models/3-5-1-2-bottle-1loss1-inlen-1-step-end-1-wc1e-4-hid-256-GRA-CLIP_model/ACC-repair-3-5-1-2-bottle-1loss1-inlen-1-step-end-1-wc1e-4-hid-256-GRA-CLIP-model_04-01-2025_09-26-00_hiddensize256'
    # epoch_idx = 4450
    # note = '3-5-1-2'

    # ## Norm all 3-11
    # parent_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/models/3-11_bottle_2LSTM_hid128_NORMALL.py_model/ACC-repair-3-11_bottle_2LSTM_hid128_NORMALL.py-model_06-01-2025_02-15-24_hiddensize32'
    # epoch_idx = 8500
    # note = '3-11'

    # ## 3-12
    parent_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/models/3-12_bottle_normal_with_3loss1_model/ACC-repair-3-12_bottle_normal_with_3loss1-model_06-01-2025_12-18-30_hiddensize128'
    epoch_idx = 6270 # 1220 1140
    note = '3-12'
    
    saved_model_dir = metric.search_model_at_epoch(parent_dir, epoch_idx)
    model_config = metric.load_model_config(saved_model_dir)
    if data_dir is not None and 'data_dir' in model_config:
        if data_dir != model_config['data_dir']:
            global_util_printer.print_yellow(f'The data_dir in the model_config is different from the data_dir in the main function. Which one should be used [1/2]?')
            print(f'1. data_dir in the main function:   {data_dir}')
            print(f'2. data_dir in the MODEL_CONFIG:    {model_config["data_dir"]}')
            choice = input()
            if choice == '1':
                pass
            elif choice == '2':
                data_dir = model_config['data_dir']
            else:
                raise ValueError('Invalid choice')
        # remove the data_dir in the model_config dictionary
        model_config.pop('data_dir')

    nae = NAEDynamicLSTM(**model_config, data_dir=data_dir, device=device)
    step_start=model_config['data_step_start']
    step_end=model_config['data_step_end']
    increment=model_config['data_increment']
            




            
    # load data
    nae_data_loader = NAEDataLoader()
    data_train_raw, data_val_raw, data_test_raw = nae_data_loader.load_train_val_test_dataset(data_dir)
    # if not nae.data_correction_check(data_train_raw, data_val_raw, data_test_raw):
    #     return
    # prepare data for training
    epoch_idx = nae.load_model(saved_model_dir, weights_only=True)

    acc_err_pairs = []
    IE_pairs = []
    for id_traj, traj in enumerate(data_test_raw):
        traj_len = len(traj['preprocess']['model_data'])
        # if id_traj != 5:
        #     continue
        '''
        Consider each trajectory
        The trajectory is split into many input-label pairs
        '''
        input_label_generator = InputLabelGenerator()

        # Pairing the trajectory into many input-label pairs
        data_test = input_label_generator.generate_input_label_dynamic_seqs([traj], step_start=step_start, step_end=step_end, increment=increment, shuffle=False)
        # Inference
        predicted_seqs, label_seqs, final_err_var_penalty = nae.validate_and_score(data=data_test, shuffle=False, inference=True)

        print('\n')
        global_util_printer.print_blue('---------------------------------------------------', background=True)
        print('Considering trajectory: ', id_traj)
        print(f'    There {len(predicted_seqs)} groups of input-label-prediction sequences')

        input_data = [inp[0] for inp in data_test]
        # convert all elements of input_data to numpy
        input_data = [inp.cpu().numpy() for inp in input_data]

        

        # 1. Calculate accumulated error for one trajectory
        eval_result = metric.compute(input_data, label_seqs, predicted_seqs)

        # print('eval_result len: ', eval_result[0].keys()); input()
        if eval_result == None:
            metric.util_printer.print_red(f'Error in metric calculation', background=True)
            return
        
        # 2.1 Show line chart of change in accumulated error with increasing input length
        # acer keys:  dict_keys(['input_len', 'len_left', 'impact_point_err', 'accumulated_error'])
        len_left = [acer['input_len'] for acer in eval_result]
        mean_acc_errs = [acer['accumulated_error'] for acer in eval_result]
        ie = [acer['impact_point_err'] for acer in eval_result]

        
        acc_err_pairs.append((len_left, mean_acc_errs))
        IE_pairs.append((len_left, ie))
            # metric.util_plotter.plot_line_chart(x_values = len_left, y_values = [mean_acc_errs], 
            #                                 x_tick_distance=5, 
            #                                 y_tick_distance=0.02,
            #                                 font_size_title=32,
            #                                 font_size_label=24,
            #                                 font_size_tick=20,
            #                                 title=f'{thrown_object} - Accumulated error by input length - Trajectory #{id_traj}', 
            #                                 x_label='Input length (data points)', 
            #                                 y_label='Accumulated error (m)',
            #                                 legends=None,
            #                                 save_plot=False)
            # 2.2 Show the predictions in 3D plot
        plot = input('    Do you want to plot [y/n] ? ')
        ## 2. Plot
        if plot=='y':
            metric.util_plotter.plot_predictions_plotly(inputs=input_data, labels=label_seqs, predictions=predicted_seqs, 
                                                        title=f'{thrown_object} - Predictions - Trajectory #{id_traj} with len {traj_len} - EPOCH {epoch_idx}', rotate_data_whose_y_up=True, 
                                                        save_plot=False, font_size_note=12,
                                                        show_all_as_default=False)
    
    metric.util_plotter.plot_variable_length_line_chart(acc_err_pairs, x_label='prediction length (data points)', y_label='Accumulated error (m)', title=f'{thrown_object} - Accumulated error by prediction length', save_plot=False)
    metric.util_plotter.plot_variable_length_line_chart(IE_pairs, x_label='prediction length (data points)', y_label='Impact point error (m)', title=f'{thrown_object} - Impact point error by prediction length', save_plot=False)
if __name__ == '__main__':
    main()