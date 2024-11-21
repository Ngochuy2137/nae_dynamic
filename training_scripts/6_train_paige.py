from nae.nae import *
from nae.utils.submodules.training_utils.data_loader import DataLoader as NAEDataLoader
from nae.utils.submodules.training_utils.input_label_generator import InputLabelGenerator
import random
DEVICE = torch.device("cuda")
def main():
    # ===================== TRAINING =====================
    # set seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    thrown_object = 'paige'
    # get dir of this script
    data_folder = os.path.dirname(os.path.realpath(__file__))
    # cd ..
    data_folder = os.path.dirname(data_folder)
    # cd to data folder
    data_folder = os.path.join(data_folder, 'data')
    data_subfolder = 'nae_paper_dataset/new_data_format/' + thrown_object
    data_folder = os.path.join(data_folder, data_subfolder)

    # Learning parameters
    input_lenth = 15
    future_pred_len = 35
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
        'hidden_size': 128,
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

    # ===================== TRAINING =====================
    print('TRAINING NAE MODEL')
    wdb_notes = f'{model_params["num_layers_lstm"]} LSTM layers, {model_params["hidden_size"]} hidden size, lr={model_params["lr"]}, batch_size={training_params["batch_size_train"]}'
    nae.init_wandb(project_name='nae',
                   run_id=wdb_run_id, 
                   resume=wdb_resume,
                   wdb_notes=wdb_notes)

    checkpoint_path = None
    saved_model_dir = nae.train(data_train, data_val, checkpoint_path=checkpoint_path)

    # # ===================== PREDICT =====================
    # print('PREDICTING')
    # # saved_model_dir = "/home/huynn/huynn_ws/edge-server-project/isaac_sim_learning/isaacsim_nae/NAE/models/21-06-2024_05-09-46-teacher-new-refactored-data/epochs5000_data100_batchsize128_seq100_pred50_timemin122-38"
    # num_trials = 3
    # # shuffle data
    # # random.shuffle(data_test)

    # inputs_test = data_test[0][:num_trials]
    # inputs_test = torch.tensor(inputs_test, dtype=torch.float32)  # Chuyển numpy array thành tensor
    # inputs_test = inputs_test.to(DEVICE)  # Chuyển tensor sang device

    # labels_test = data_test[1][:num_trials, 1:, :]
    # predictions_test = nae.predict(inputs_test).cpu().numpy()

    # # scoring
    # mean_mse_all, mean_mse_xyz, mean_ade, \
    # (mean_nade, var_nade), (mean_fe, var_fe), \
    # mean_nade_future, capture_success_rate = nae.utils.score_all_predictions(predictions_test, 
    #                                                                             labels_test, 
    #                                                                             future_pred_steps = inputs_test.shape[1],
    #                                                                             capture_thres=0.1)
    # print('\nPrediction score:')
    # # print(' Mean MSE all            : ', mean_mse_all)
    # # print(' Mean MSE xyz            : ', mean_mse_xyz )
    # # print(' Mean ADE             (m): ', mean_ade)
    # # print(' Mean NADE               : ', mean_nade)
    # # print(' Mean NADE future        : ', mean_nade_future)
    # print('     Mean final step err (m) : ', mean_fe)
    # print('     Capture success rate    : ', capture_success_rate)
    # print('     trials: ', num_trials)

    # # plotting
    # plot_title = 'Prediction'
    # nae.utils.plotter.plot_predictions(predictions_test, labels_test, title=plot_title)

if __name__ == '__main__':
    main()