from nae_core.evaluation.NAE_metrics.metric import *
from python_utils.printer import Printer
import glob

global_printer = Printer()
def main():
    metric = MetricGoalError()
    device = torch.device('cuda')
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # --- Data and model ---
    id_traj = 'last 70 frames'
    filter_key = 'len_left'
    filter_value = 'goal_error'
    # ## ----------------- 1. Bamboo -----------------
    # data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/data/nae_paper_dataset/data_preprocessed/Bamboo'
    # thrown_object = 'bamboo'
    # parent_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/models/ACC-repair-bamboo_model/bamboo-model_01-01-2025_06-58-11_hiddensize128'
    # epoch_idx = 1220 # 1210 1220
    # saved_model_dir = glob.glob(f'{parent_dir}/*epochs{epoch_idx}*')[0]


    # ## ----------------- 3. Bottle -----------------\
    data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/data/nae_paper_dataset/data_preprocessed/Bottle'
    thrown_object = 'bottle'
    # # ----- lr = 5e-5
    # parent_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/models/ACC-repair-bottle-lr5e-5_model/bottle-lr5e-5-model_01-01-2025_22-28-52_hiddensize128'
    # epoch_idx = 5530 #6900
    # note = 'lr: 5e-5'

    ## ----- lr = 1e-4
    # parent_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/models/ACC-repair-bottle-lr1e-4_model/bottle-lr1e-4-model_01-01-2025_23-58-41_hiddensize128'
    # epoch_idx = 5360    #   3600 4680 5360
    # note = 'lr: 1e-4'

    # ## ----- 2.0 * Loss1
    # parent_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/models/ACC-repair-bottle-2loss1_model/bottle-2loss1-model_02-01-2025_02-26-46_hiddensize128'
    # epoch_idx = 1180
    # note = '2.0 * Loss1'

    ## ----- 3.0 * Loss1
    parent_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/models/ACC-repair-bottle-3loss1_model/bottle-3loss1-model_02-01-2025_12-04-38_hiddensize128'
    epoch_idx = 4600
    note = '3.0 * Loss1'
    
    
    saved_model_dir = glob.glob(f'{parent_dir}/*epochs{epoch_idx}*')[0]
    # Training parameters 
    training_params = {
        'num_epochs': 5000,
        'batch_size_train': 128,    
        'batch_size_val': 1024,
        'save_interval': 10,
        'thrown_object' : thrown_object + '-dynamic-len',
        'train_id': None
    }
    # Model parameters
    model_params = {
        'input_size': 9,
        'hidden_size': 128,
        'output_size': 9,
        'num_layers_lstm': 2,
        'lr': 0.0001,
        'dropout_rate': 0.0
    }

    nae = NAEDynamicLSTM(**model_params, **training_params, data_dir=data_dir, device=device)
    # load data
    nae_data_loader = NAEDataLoader()
    data_train_raw, data_val_raw, data_test_raw = nae_data_loader.load_train_val_test_dataset(data_dir)
    # prepare data for training
    epoch_idx = nae.load_model(saved_model_dir, weights_only=True)
    if epoch_idx is None:
        raise ValueError('Cannot determine the epoch index')
    '''
    Consider all trajectory in the test dataset
    The trajectory is split into many input-label pairs
    '''
    input_label_generator = InputLabelGenerator()
    # Pairing the trajectory into many input-label pairs
    data_test = input_label_generator.generate_input_label_dynamic_seqs(data_test_raw, step_start=5, step_end=-3, increment=1, shuffle=False)
    # Inference
    predicted_seqs, label_seqs, final_err_var_penalty = nae.validate_and_score(data=data_test, batch_size=1024, shuffle=False, inference=True)

    print(f'There {len(predicted_seqs)} groups of input-label-prediction sequences')
    global_printer.print_green(f'final_err_var_penalty: {final_err_var_penalty}', background=True)

    input_seqs = [inp[0] for inp in data_test]
    metric.process_and_plot(input_seqs=input_seqs, predicted_seqs=predicted_seqs, label_seqs=label_seqs, 
                            thrown_object=thrown_object, id_traj=id_traj, 
                            filter_key=filter_key, filter_value=filter_value, epoch_idx=epoch_idx, note=note)
    

if __name__ == '__main__':
    main()