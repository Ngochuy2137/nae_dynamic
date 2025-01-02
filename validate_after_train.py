from nae_core.evaluation.NAE_metrics.metric import *
from python_utils.printer import Printer
import glob
import os
import re
import wandb
import subprocess
# import lib to show progress bar
from tqdm import tqdm

global_util_printer = Printer()

parent_folder = "/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/models/ACC-repair-bottle-2loss1_model/bottle-2loss1-model_02-01-2025_09-15-20_hiddensize128"

def list_all_subfolders(directory):
    subfolders = []
    for root, dirs, _ in os.walk(directory):
        for dir_name in dirs:
            subfolders.append(os.path.join(root, dir_name))
    return subfolders

def extract_epoch_and_path(subfolders):
    extracted_data = []
    for folder in subfolders:
        folder_name = os.path.basename(folder)  # Lấy tên thư mục
        # Tìm số epoch từ tên thư mục (format epochsXXXX_)
        match = re.search(r'epochs(\d+)', folder_name)
        if match:
            epoch = int(match.group(1))  # Lấy số epoch
            extracted_data.append((epoch, folder))  # Lưu (epoch, đường dẫn)
    return extracted_data

def get_model_epochs_and_paths(parent_folder):
    subfolders = list_all_subfolders(parent_folder)
    # Trích xuất thông tin epoch và đường dẫn
    epoch_and_paths = extract_epoch_and_path(subfolders)
    sorted_epoch_and_paths = sorted(epoch_and_paths, key=lambda x: x[0])
    return sorted_epoch_and_paths

def main():
    metric = MetricGoalError()
    device = torch.device('cuda')
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/data/nae_paper_dataset/data_preprocessed/Bottle'
    ## ----------------- 2*loss1 -----------------\
    # thrown_object = 'bottle-2loss1'
    # model_parent_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/models/ACC-repair-bottle-2loss1_model/bottle-2loss1-model_02-01-2025_09-15-20_hiddensize128'
    
    ## ----------------- 3*loss1 -----------------\
    thrown_object = 'bottle-3loss1'
    model_parent_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/models/ACC-repair-bottle-3loss1_model/bottle-3loss1-model_02-01-2025_12-04-38_hiddensize128'


    wdb_run_id=None   # 't5nlloi0'
    wdb_resume=None   # 'allow'
    enable_wandb = True

    # Training parameters 
    training_params = {
        'num_epochs': 12000,
        'batch_size_train': 512,    
        'batch_size_val': 1024,
        'save_interval': 10,
        'thrown_object' : thrown_object,
        'train_id': 'NEW METRIC',
        'warmup_steps': 25,
        'dropout_rate': 0.0,
        'loss1_weight': 2.0,
        'loss2_weight': 1.0,
        'loss2_1_weight': 0.0,
        'weight_decay': 0.0001,
    }
    # Model parameters
    model_params = {
        'input_size': 9,
        'hidden_size': 128,
        'output_size': 9,
        'num_layers_lstm': 2,
        'lr': 0.0001
    }
    wdb_notes = f'lr: {model_params["lr"]}, \
                {training_params["loss2_weight"]}*L2: ,  \
                {training_params["loss2_1_weight"]}*L2_1, \
                warmup {training_params["warmup_steps"]}, \
                dropout: {training_params["dropout_rate"]}, \
                weight_decay: {training_params["weight_decay"]}'

    nae = NAEDynamicLSTM(**model_params, **training_params, data_dir=data_dir, device=device)
    # load data
    nae_data_loader = NAEDataLoader()
    data_train, data_val, data_test = nae_data_loader.load_train_val_test_dataset(data_dir)

    # prepare data for training
    input_label_generator = InputLabelGenerator()
    data_train = input_label_generator.generate_input_label_dynamic_seqs(data_train, step_start=5, step_end=-3, increment=1, shuffle=True)
    data_val = input_label_generator.generate_input_label_dynamic_seqs(data_val, step_start=5, step_end=-3, increment=1, shuffle=True)
    data_test = input_label_generator.generate_input_label_dynamic_seqs(data_test, step_start=5, step_end=-3, increment=1, shuffle=True)

    print('     ----- After generating inputs, labels -----')
    print('     Training data:      ', len(data_train))
    print('     Validation data:    ', len(data_val))
    print('     Testing data:       ', len(data_test))
    print('     ----------------\n')

    nae.util_printer.print_green('Start training ...', background=True)
    if enable_wandb:
        nae.init_wandb('nae-dynamic', run_id=wdb_run_id, resume=wdb_resume, wdb_notes=wdb_notes)
    
    # Load all models
    model_info = get_model_epochs_and_paths(model_parent_dir)
    if len(model_info) == 0:
        global_util_printer.print_red('No model found')
        raise ValueError('No model found')
    
    # validation
    # show progress bar
    for epoch_idx, model_dir in tqdm(model_info):
        tqdm.write(f"Epoch {epoch_idx}: Processing model at {model_dir}")
        if nae.load_model(model_dir, weights_only=True) != epoch_idx:
            raise ValueError('Cannot determine the epoch index')
        # 2. ----- FOR VALIDATION -----
        # logging.info(f"VALIDATION:")
        mean_loss_total_val_log, \
        (mean_ade_entire,   std_ade_entire), \
        (mean_ade_future,   std_ade_future), \
        (mean_ade_past,     std_ade_past), \
        (mean_nade_entire,  std_nade_entire), \
        (mean_nade_future,  std_nade_future), \
        (mean_nade_past,    std_nade_past), \
        (mean_fe, std_fe, converge_to_final_point_trending), \
        capture_success_rates = nae.validate_and_score(data=data_val, shuffle=False)
        
        # input('Say hahaha')
        # 3. ----- FOR WANDB LOG -----
        if enable_wandb:
            wandb.log({
                'valid_loss_total': mean_loss_total_val_log,
            }, step=epoch_idx)

            wandb.log({
                'valid_mean_ade_entire': mean_ade_entire,
                'valid_mean_ade_entire_min': mean_ade_entire - std_ade_entire,
                'valid_mean_ade_entire_max': mean_ade_entire + std_ade_entire,
            }, step=epoch_idx)

            wandb.log({
                'valid_mean_ade_future': mean_ade_future,
                'valid_mean_ade_future_min': mean_ade_future - std_ade_future,
                'valid_mean_ade_future_max': mean_ade_future + std_ade_future,
            }, step=epoch_idx)

            wandb.log({
                'valid_mean_ade_past': mean_ade_past,
                'valid_mean_ade_past_min': mean_ade_past - std_ade_past,
                'valid_mean_ade_past_max': mean_ade_past + std_ade_past,
            }, step=epoch_idx)

            wandb.log({
                'valid_mean_nade_entire': mean_nade_entire,
                'valid_mean_nade_entire_min': mean_nade_entire - std_nade_entire,
                'valid_mean_nade_entire_max': mean_nade_entire + std_nade_entire,
            }, step=epoch_idx)

            wandb.log({
                'valid_mean_nade_future': mean_nade_future,
                'valid_mean_nade_future_min': mean_nade_future - std_nade_future,
                'valid_mean_nade_future_max': mean_nade_future + std_nade_future,
            }, step=epoch_idx)

            wandb.log({
                'valid_mean_nade_past': mean_nade_past,
                'valid_mean_nade_past_min': mean_nade_past - std_nade_past,
                'valid_mean_nade_past_max': mean_nade_past + std_nade_past,
            }, step=epoch_idx)

            wandb.log({
                "valid_mean_IE_err": mean_fe,
                "valide_mean_IE_min": mean_fe - std_fe,
                "valid_mean_IE_max": mean_fe + std_fe,
            }, step=epoch_idx)

            for cap_rate_data in capture_success_rates:
                cap_thr = cap_rate_data[0]
                cap_rate = cap_rate_data[1]
                cap_std = cap_rate_data[2]
                wandb.log({
                    f"valid_capture_success_rate_{cap_thr}": cap_rate,
                    f"valid_capture_success_rate_{cap_thr}_min": cap_rate - cap_std,
                    f"valid_capture_success_rate_{cap_thr}_max": cap_rate + cap_std,
                }, step=epoch_idx)

            wandb.log({
                'converge_to_final_point_trending': converge_to_final_point_trending,
            }, step=epoch_idx)


if __name__ == '__main__':
    main()