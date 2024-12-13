from nae_core.evaluation.NAE_metrics.metric import *

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

    # data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/data/nae_paper_dataset/new_data_format/bamboo/split/bamboo'
    # object_name = 'bamboo'
    # # saved_model_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/models/bamboo-dynamic-len_model/NAE_DYNAMIC-model_09-12-2024_19-35-08_hiddensize128/@epochs500_data14770_batchsize256_hiddensize128_timemin108-23_NAE_DYNAMIC'
    # saved_model_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/models/bamboo-dynamic-len_model/NAE_DYNAMIC-model_09-12-2024_19-35-08_hiddensize128/@epochs2880_data14770_batchsize256_hiddensize128_timemin626-51_NAE_DYNAMIC'

    # data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/data/nae_paper_dataset/new_data_format/banana/split/banana'
    # thrown_object = 'banana'
    # # saved_model_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/models/banana-dynamic-len_model/NAE_DYNAMIC-model_10-12-2024_19-59-34_hiddensize128/@epochs450_data55011_batchsize256_hiddensize128_timemin211-94_NAE_DYNAMIC'
    # saved_model_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/models/banana-dynamic-len_model/NAE_DYNAMIC-model_10-12-2024_19-59-34_hiddensize128/@epochs880_data55011_batchsize256_hiddensize128_timemin504-96_NAE_DYNAMIC'

    # data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/data/nae_paper_dataset/new_data_format/bottle/split/bottle'
    # thrown_object = 'bottle'
    # saved_model_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/models/bottle-dynamic-len_model/NAE_DYNAMIC-model_11-12-2024_14-09-03_hiddensize128/@epochs2660_data9262_batchsize256_hiddensize128_timemin312-51_NAE_DYNAMIC'

    data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/data/nae_paper_dataset/new_data_format/gourd/split/gourd'
    thrown_object = 'gourd'
    saved_model_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/models/gourd-dynamic-len_model/NAE_DYNAMIC-model_11-12-2024_23-57-46_hiddensize128/@epochs520_data18616_batchsize256_hiddensize128_timemin133-66_NAE_DYNAMIC'
    # saved_model_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/models/gourd-dynamic-len_model/NAE_DYNAMIC-model_12-12-2024_13-28-27_hiddensize128-warmup/@epochs810_data18616_batchsize256_hiddensize128_timemin214-54_NAE_DYNAMIC'
    # saved_model_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/models/gourd-dynamic-len_model/NAE_DYNAMIC-model_13-12-2024_12-28-33_hiddensize128-warmup-weightdecay/@epochs840_data18616_batchsize256_hiddensize128_timemin221-04_NAE_DYNAMIC'


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

    input_seqs = [inp[0] for inp in data_test]
    metric.process_and_plot(input_seqs=input_seqs, predicted_seqs=predicted_seqs, label_seqs=label_seqs, 
                            thrown_object=thrown_object, id_traj=id_traj, 
                            filter_key=filter_key, filter_value=filter_value)
    

if __name__ == '__main__':
    main()