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
    ## ----------------- 1. Bamboo -----------------
    # data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/data/nae_paper_dataset/new_data_format/bamboo/split/bamboo'
    # thrown_object = 'bamboo'
    # parent_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/models/Done-Finetune-bamboo_model/after-finetune-gourd-model_22-12-2024_17-20-21_hiddensize128'
    # epoch_idx = 2070 # 880 1510 1550 1960 2070
    # saved_model_dir = glob.glob(f'{parent_dir}/*epochs{epoch_idx}*')[0]



    ## ----------------- 3. gourd -----------------
    # data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/data/nae_paper_dataset/new_data_format/gourd/split/gourd'
    # thrown_object = 'gourd'
    # parent_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/models/backup/gourd-TORCH21-NEW_model/0-model_21-12-2024_04-57-06_hiddensize128-origin'
    # epoch_idx = 0
    # saved_model_dir = glob.glob(f'{parent_dir}/*epochs{epoch_idx}*')[0]


    ## ----------------- 4. Green -----------------
    data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/data/nae_paper_dataset/new_data_format/green/split/green'
    thrown_object = 'green'
    parent_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/models/Done-Finetune-green_model/green-model_23-12-2024_19-41-52_hiddensize128'
    epoch_idx = 890      # 890 1040
    saved_model_dir = glob.glob(f'{parent_dir}/*epochs{epoch_idx}*')[0]
    




    # ----------------- 6. Paige -----------------
    # data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/data/nae_paper_dataset/new_data_format/paige/split/paige'
    # thrown_object = 'paige'
    # saved_model_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/models/paige-dynamic-len_model/NAE_DYNAMIC-model_15-12-2024_16-01-02_hiddensize128/epochs940_data15761_batchsize256_hiddensize128_timemin27-9_NAE_DYNAMIC'


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
    data_train_raw, data_val_raw, data_test_raw = nae_data_loader.load_dataset(data_dir)
    if not nae.data_correction_check(data_train_raw, data_val_raw, data_test_raw):
        return
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
                            filter_key=filter_key, filter_value=filter_value, epoch_idx=epoch_idx)
    

if __name__ == '__main__':
    main()