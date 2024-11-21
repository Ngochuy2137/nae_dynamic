from nae.nae import *
from nae.utils.submodules.training_utils.data_loader import DataLoader as NAEDataLoader
from nae.utils.submodules.training_utils.input_label_generator import InputLabelGenerator
import random
DEVICE = torch.device("cuda")
def main():
    # set seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    thrown_object = 'bamboo'
    data_folder = 'data/nae_paper_dataset/split/2024-09-27'
    # data_folder = os.path.join(data_folder, thrown_object)
    data_folder = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_prediction_ws/src/nae/data/frisbee-pbl'

    # Learning parameters
    input_lenth = 10
    future_pred_len = 15
    training_params = {
        'input_len': input_lenth,
        'future_pred_len': future_pred_len,
        'num_epochs': 5000,
        'batch_size_train': 128,    
        'batch_size_val': 32,
        'save_interval': 10,
        'thrown_object' : thrown_object + '-input-' + str(input_lenth) + 'pred-' + str(future_pred_len)
    }
    # Wandb parameters
    wdb_run_id=None   # 't5nlloi0'
    wdb_resume=None   # 'allow'
    model_params = {
        'input_size': 9,
        'hidden_size': 256,
        'output_size': 9,
        'num_layers_lstm': 2,
        'lr': 0.0001
    }

    nae = NAE(**model_params, **training_params, data_dir=data_folder, device=DEVICE)
    # load data
    nae_data_loader = NAEDataLoader()
    data_train, data_val, data_test = nae_data_loader.load_dataset(data_folder)

    
    print('     ----- Before generating inputs, labels -----')
    print('     Data dir:      ', data_folder)
    print('     Training data:      ', len(data_train))
    print('     Validation data:    ', len(data_val))
    print('     Testing data:       ', len(data_test))
    print('     input_len:         ', training_params['input_len'])
    print('     future_pred_len:   ', training_params['future_pred_len'])
    print('     ----------------\n')

    # generate input and label sequences
    input_label_generator = InputLabelGenerator()
    data_train  = input_label_generator.generate_input_label_seq(data_train, 
                                                                training_params['input_len'], 
                                                                training_params['future_pred_len'])
    data_val    = input_label_generator.generate_input_label_seq(data_val, 
                                                                training_params['input_len'], 
                                                                training_params['future_pred_len'])
    data_test   = input_label_generator.generate_input_label_seq(data_test,
                                                                training_params['input_len'], 
                                                                training_params['future_pred_len'])
    
    print('     ----- After generating inputs, labels -----')
    print('     Training data:      ', data_train[0].shape, ' ', data_train[1].shape)
    print('     Validation data:    ', data_val[0].shape, ' ', data_val[1].shape)
    print('     Testing data:       ', data_test[0].shape, ' ', data_test[1].shape)
    print('     ----------------\n')

    # test_plot = [data_val[0][15], data_val[1][15]]
    # nae.utils.plotter.plot_samples(test_plot)
    # input('Press Enter to train the model')

    # # ===================== TRAINING =====================
    # print('TRAINING NAE MODEL')
    # nae.init_wandb(project_name='nae',
    #                run_id=wdb_run_id, 
    #                resume=wdb_resume)

    # checkpoint_path = None
    # saved_model_dir = nae.train(data_train, data_val, checkpoint_path=checkpoint_path)

    # ===================== PREDICT =====================
    print('PREDICTING')
    saved_model_path = "/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_prediction_ws/src/nae/training_scripts/models/frisbee-pbl-input-15pred-50_model/run_NAE3L_14-11-2024_06-00-07/@epochs560_data2_batchsize256_seq15_pred50_timemin2-51"
    capture_radius = 0.1
    plot = True

    # 1. load data 
    print('    Loading data: ')
    input_seqs = data_test[0][:, :, :]
    label_seqs = data_test[1][:, 1:, :]

    # 2. load model
    print('    Loading model: ', saved_model_path)
    nae.load_model(saved_model_path)
    
    # 3. Predict
    prediction_time = time.time()
    print('\n--- PREDICTING ---')
    predicted_seqs = nae.predict(input_seqs, evaluation=True).cpu().numpy()
    prediction_time = time.time() - prediction_time
    print('     [TIME] Prediction time: ', prediction_time)
    rate = 1 / prediction_time
    print('     [RATE] Prediction rate: ', 1 / prediction_time)

    # n = random.randint(0, len(input_seqs))
    # print('\n     check input_len       :', len(input_seqs[n]))
    # print('     check future_pred_len :', len(predicted_seqs[n]))


    # 4. scoring
    mean_mse_all, mean_mse_xyz, mean_ade, \
    (mean_nade, var_nade), (mean_fe, var_fe), \
    mean_nade_future, capture_success_rate = nae.utils.score_all_predictions(predicted_seqs, 
                                                                                label_seqs, 
                                                                                training_params['future_pred_len'],
                                                                                capture_thres=capture_radius)
    print('\nPrediction score:')
    print('     Object: ', thrown_object)
    print('     Model: ', saved_model_path)
    # print(' Mean MSE all            : ', mean_mse_all)
    # print(' Mean MSE xyz            : ', mean_mse_xyz )
    # print(' Mean ADE             (m): ', mean_ade)
    # print(' Mean NADE               : ', mean_nade)
    # print(' Mean NADE future        : ', mean_nade_future)
    print('     Mean final step err (m) : ', mean_fe)
    print('     Capture success rate    : ', capture_success_rate)
    num_trials = len(input_seqs)
    print('     trials: ', num_trials)

    # plotting
    if plot:
        plot_title = 'object ' + thrown_object
        nae.utils.plotter.plot_predictions(predicted_seqs, label_seqs, lim_plot_num=5, swap_y_z=True, title=plot_title)

if __name__ == '__main__':
    main()