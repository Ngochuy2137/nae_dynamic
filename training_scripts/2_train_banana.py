from nae_core.nae_dynamic import *
from nae_core.utils.submodules.training_utils.data_loader import DataLoader as NAEDataLoader
from nae_core.utils.submodules.training_utils.input_label_generator import InputLabelGenerator
import random

def main():
    device = torch.device('cuda')
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/data/nae_paper_dataset/new_data_format/banana/split/banana'
    thrown_object = 'banana'
    
    checkout_path = None
    wdb_run_id=None   # 't5nlloi0'
    wdb_resume=None   # 'allow'
    enable_wandb = True

    # Training parameters 
    training_params = {
        'num_epochs': 2500,
        'batch_size_train': 512,    
        'batch_size_val': 1024,
        'save_interval': 10,
        'thrown_object' : 'Done-Finetune-' + thrown_object,
        'train_id': 'after-finetune-gourd',
        'warmup_steps': 25,
        'dropout_rate': 0.0,
        'loss2_weight': 1.0,
        'loss2_1_weight': 0.0,
        'weight_decay': 0.0001,
        'lstm_bidirectional': False
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
                weight_decay: {training_params["weight_decay"]}, \
                lstm_bidirectional: {training_params["lstm_bidirectional"]}'

    nae = NAEDynamicLSTM(**model_params, **training_params, data_dir=data_dir, device=device)
    # load data
    nae_data_loader = NAEDataLoader()
    data_train, data_val, data_test = nae_data_loader.load_dataset(data_dir)
    if not nae.data_correction_check(data_train, data_val, data_test):
        return
    
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


    # input('DEBUG')
    # data_train = data_train[:128]
    # 2. Training
    # data_train = data_train[:128]
    # data_val = data_val[:128]
    nae.util_printer.print_green('Start training ...', background=True)
    if enable_wandb:
        nae.init_wandb('nae-dynamic', run_id=wdb_run_id, resume=wdb_resume, wdb_notes=wdb_notes)
    saved_model_dir = nae.train(data_train, data_val, checkpoint_path=checkout_path, enable_wandb=enable_wandb, 
                                test_anomaly=False, 
                                test_cuda_blocking=False)

if __name__ == '__main__':
    main()