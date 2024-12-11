from nae_core.nae_dynamic import NAEDynamicLSTM
from nae_core.utils.submodules.training_utils.data_loader import DataLoader as NAEDataLoader
from nae_core.utils.submodules.training_utils.input_label_generator import InputLabelGenerator
from .metric import Metric

from python_utils.plotter import Plotter
from python_utils.printer import Printer

import numpy as np
import torch
import random


class MetricAccumulatedError(Metric):
    def __init__(self):
        self.util_printer = Printer()
        self.util_plotter = Plotter()
        
    def compute(self, input_seqs, label_seqs, predicted_seqs):
        '''
        We will examine how the accumulated error changes with increasing input length
        The input data length is increased by 1 data point each time
        The input_seqs includes input seqs with increasing length
        (We will get mean accumulated error for each input length)
        '''
        # 0. check if the prediction is proper:
        if not self.is_proper_prediction(input_seqs, label_seqs):
            self.util_printer.print_red(f'{len(input_seqs)} labels are incorrect')
            return
        self.util_printer.print_green(f'{len(input_seqs)} labels are correct')

        # 1. Calculate accumulated error for each prediction
        accumulated_error_by_input_length = []
        last_err_list = []
        # count = 0

        # Consider one group (input, label, predicted) at a time
        for inp, pred, lab in zip(input_seqs, predicted_seqs, label_seqs):
            # Only calculate the accumulated error for the first 3 dimensions x, y, z
            inp = inp[:, :3]
            lab = lab[:, :3]
            pred = pred[:, :3]
            # input('check pred, lab shape: ' + str(pred.shape) + ' ' + str(lab.shape))

            dis = np.linalg.norm(pred - lab, axis=-1)
            accumulated_error = np.mean(dis)
            pred_last = pred[-1]
            lab_last = lab[-1]
            last_err = np.linalg.norm(pred_last - lab_last)
            last_err_list.append(last_err)

            err_by_inlen = {
                'input_len': len(inp),
                'len_left': len(lab) - len(inp),
                'accumulated_error': accumulated_error
            }
            accumulated_error_by_input_length.append(err_by_inlen) 
        return accumulated_error_by_input_length
    


def main():
    metric = MetricAccumulatedError()
    device = torch.device('cuda')
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # feature_size = 9
    # # 1. Dataset and DataLoader
    # dataset = TimeSeriesDataset(num_samples=100, max_len=15, feature_size=feature_size)

    data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_prediction_ws/src/nae/data/rllab_dataset_no_orientation/data_enrichment/big_plane/big_plane_enrich_for_training'
    thrown_object = 'big_plane'

    # data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_prediction_ws/src/nae/data/nae_paper_dataset/new_data_format/bamboo/split/bamboo'
    # thrown_object = 'bamboo'

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
    saved_model_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/models/big_plane-dynamic-len_model/NAE_DYNAMIC-model_02-12-2024_19-10-47_hiddensize128/@epochs260_data31529_batchsize128_hiddensize128_timemin195-45_NAE_DYNAMIC'
    nae.load_model(saved_model_dir, weights_only=True)

    # def evaluate_one_trajectory_prediction(self, trajectory):
    for id, traj in enumerate(data_test_raw):
        '''
        Consider each trajectory
        The trajectory is split into many input-label pairs
        '''

        input_label_generator = InputLabelGenerator()

        # Pairing the trajectory into many input-label pairs
        data_test = input_label_generator.generate_input_label_dynamic_seqs([traj], step_start=5, step_end=-3, increment=1, shuffle=False)
        # Inference
        predicted_seqs, label_seqs = nae.validate_and_score(data=data_test, batch_size=1024, shuffle=False, inference=True)

        print(f'There {len(predicted_seqs)} groups of input-label-prediction sequences')

        input_data = [inp[0] for inp in data_test]
        # convert all elements of input_data to numpy
        input_data = [inp.cpu().numpy() for inp in input_data]

        

        # Calculate accumulated error for one trajectory
        accumulated_err = metric.compute(input_data, label_seqs, predicted_seqs, thrown_object, id, plot=False)
        if accumulated_err == None:
            metric.util_printer.print_red(f'Error in accumulated error calculation', background=True)
            return
        
        input('Do you want to check next trajectory?')
        
        # def nae_metrics(self, predicted_seqs, label_seqs):
        #     # Calculate metrics

        # def metric_leading_time(self, predicted_seqs, label_seqs, error_tar):
        #     # Calculate leading time

        
                                    
    input()
    

if __name__ == '__main__':
    main()