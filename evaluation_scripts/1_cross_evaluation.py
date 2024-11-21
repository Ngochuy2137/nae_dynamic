from nae.nae import *
from nae.utils.submodules.training_utils.data_loader import DataLoader as NAEDataLoader
from nae.utils.submodules.training_utils.input_label_generator import InputLabelGenerator
from nae.utils.submodules.training_utils.evaluator import Evaluator
import random
import math
from collections import defaultdict
import json
import time
import pandas as pd

DEVICE = torch.device("cuda")

def main():
    start_time = time.time()
    evaluator = Evaluator()
    data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae/data/nae_paper_dataset/split/2024-09-27'
    model_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae/training_scripts/models/input-10pred-15'

    capture_radius = 0.1
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
        'thrown_object' : 'cross-evaluation' + '-input-' + str(input_lenth) + 'pred-' + str(future_pred_len)
    }

    model_params = {
        'input_size': 9,
        'hidden_size': 128,
        'output_size': 9,
        'num_layers_lstm': 2,
        'lr': 0.0001
    }

    # 1. load object data
    object_paths = {
        'bamboo' : data_dir + '/bamboo',
        'banana' : data_dir + '/banana',
        'bottle' : data_dir + '/bottle',
        'gourd' : data_dir + '/gourd',
        'green' : data_dir + '/green',
        'paige' : data_dir + '/paige'
    }
    nae_data_loader = NAEDataLoader()
    objects_data = {}
    for obj in object_paths.items():
        data_path = obj[1]
        _, _, data_test = nae_data_loader.load_dataset(data_path)
        # generate input and label sequences
        input_label_generator = InputLabelGenerator()
        input_seqs, label_seqs = input_label_generator.generate_input_label_seq(data_test,
                                                                                training_params['input_len'], 
                                                                                training_params['future_pred_len'])
        objects_data[obj[0]] = {}
        objects_data[obj[0]]['input_seqs'] = input_seqs
        objects_data[obj[0]]['label_seqs'] = label_seqs



    # bamboo_path = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae/data/nae_paper_dataset/split/2024-09-27/bamboo/data_train_len_142.npy'
    # nae_data_loader = NAEDataLoader()
    # # _, _, data_test = nae_data_loader.load_dataset(bamboo_path)
    # data_test = np.load(bamboo_path, allow_pickle=True)
    # print('check bamboo data test: ', len(data_test))
    # input()
    # input_label_generator = InputLabelGenerator()
    # input_seqs, label_seqs = input_label_generator.generate_input_label_seq(data_test,
    #                                                                         training_params['input_len'], 
    #                                                                         training_params['future_pred_len'])
    # print('check bamboo sub data test: ', len(input_seqs), ' ', len(label_seqs))
    # input()
    # objects_data = {'bamboo': 
    #                     {'input_seqs': input_seqs,
    #                     'label_seqs': label_seqs}
    #                 }

    # check objecy_info
    print('Object info:')
    for obj in objects_data.items():
        print('     ', obj[0], len(obj[1]['input_seqs']), ' ', len(obj[1]['label_seqs']))
    input('Press Enter to continue\n')

    # 2. Load models
    print('Looking for best models...')
    models_info = evaluator.find_best_models(model_dir)
    # models_info = {'bamboo': '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae/training_scripts/models/storage/bamboo-input-10pred-15/run_NAE3L_26-10-2024_13-32-37/@epochs730_data2_batchsize128_seq10_pred15_timemin13-34'}
    print('Best models:')
    for md in models_info.items():
        print('     ', md[0], ' ', md[1])
    input('Press Enter to continue\n')

    mean_final_err_scores = defaultdict(dict)
    capture_success_rate_scores = defaultdict(dict)
    for obj in objects_data.items():
        one_object_data = {'type': obj[0], 'data': obj[1]}
        for md in models_info.items():
            one_model_info = {'type': md[0], 'path': md[1]}
            _, _, mean_fe, capture_success_rate = evaluator.evaluate(one_object_data, one_model_info, model_params, training_params, capture_radius, plot=False)
            
            # Clean up a bit before saving
            mean_fe = round(mean_fe[0], 4), round(mean_fe[1], 4)
            mean_fe = f"{mean_fe[0]:.4f} ± {math.sqrt(mean_fe[1]):.4f}"
            capture_success_rate = f"{capture_success_rate:.4f}"

            mean_final_err_scores[one_object_data['type']][one_model_info['type']] = mean_fe
            capture_success_rate_scores[one_object_data['type']][one_model_info['type']] = capture_success_rate
    scores = {'mean_fe': mean_final_err_scores, 'capture_success_rate': capture_success_rate_scores}
    
    print('\n---------------------------------------------------------------------------------------------------')
    print('Mean final dist scores:')
    print(mean_final_err_scores)
    print('Capture success rate scores:')
    print(capture_success_rate_scores)

    # save results
    # get path of this script
    this_path = os.path.dirname(os.path.abspath(__file__))
    result_json_path = os.path.join(this_path, 'evaluation_scores.json')
    evaluator.save_results(scores, result_json_path)
    result_excel_path = os.path.join(this_path, 'evaluation_scores.xlsx')
    evaluator.save_results_to_excel(scores, result_excel_path)

    print('\nEvaluation result was saved to file: evaluation_scores.json')
    print("--- Evaluation time: %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()