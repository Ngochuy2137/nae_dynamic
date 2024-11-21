import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import time
import os
from datetime import datetime
import wandb
import random

from nae.utils.utils import NAE_Utils
from nae.utils.submodules.training_utils.data_loader import DataLoader as NAEDataLoader
from nae.utils.submodules.training_utils.input_label_dynamic_generator import InputLabelDynamicGenerator

DEVICE = torch.device("cuda")

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.fc(x)
        out = self.tanh(out)
        return out

class LSTMModel(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.lstm = nn.LSTM(hidden_size, output_size, num_layers=self.num_layers, batch_first=True)

    def forward(self, x, hi, ci):
        out, (hn, cn) = self.lstm(x, (hi, ci))
        return out, hn, cn

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc(x)
        return out
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
 
class NAE:
    def __init__(self, input_size, hidden_size, output_size, num_layers_lstm, lr,
                 input_len, future_pred_len, num_epochs, batch_size_train, batch_size_val, save_interval, thrown_object,
                 device,
                 data_dir=''):
        
        self.utils = NAE_Utils()
        self.device = device
        # model architecture params
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers_lstm
        self.lr = lr

        # training params
        self.t_value = input_len
        self.k_value = future_pred_len
        self.num_epochs = num_epochs
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.save_interval = save_interval
        self.thrown_object = thrown_object + '_model'
        self.data_dir = data_dir

        self.run_name = f"model_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}_hiddensize{hidden_size}_inseq{input_len}_outpred{future_pred_len}"
        self.model_dir = os.path.join('models', self.thrown_object, self.run_name)

        # Init model
        self.encoder = Encoder(input_size, hidden_size)
        self.lstm = LSTMModel(hidden_size, hidden_size, num_layers_lstm)
        self.decoder = Decoder(hidden_size, output_size)

        self.encoder.to(self.device)
        self.lstm.to(self.device)
        self.decoder.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.lstm.parameters()) + list(self.decoder.parameters()), lr=lr)

        self.plot_data_loss1 = []
        self.plot_data_loss2 = []
        self.plot_data_loss3 = []
        self.plot_data_loss_total = []

        print('\n-----------------------------------')
        print('Parameters number of encoder: ', count_parameters(self.encoder))
        print('Parameters number of LSTM: ', count_parameters(self.lstm))
        print('Parameters number of decoder: ', count_parameters(self.decoder))
        print('Total number of parameters: ', count_parameters(self.encoder) + count_parameters(self.lstm) + count_parameters(self.decoder))

    def init_wandb(self, project_name, run_id=None, resume=None, wdb_notes=''):        
        wandb.init(
            # set the wandb project where this run will be logged
            project = project_name,
            name=self.thrown_object + '_' + self.run_name,
            # id='o5qeq1n8', resume='allow',
            id=run_id, resume=resume,
            # track hyperparameters and run metadata
            config={
            "learning_rate": self.lr,
            "architecture": "LSTM",
            "dataset": self.data_dir,
            "epochs": self.num_epochs,
            "batch_size": self.batch_size_train,
            "hidden_size": self.hidden_size,
            "num_layers_lstm": self.num_layers,
            "past length": self.t_value,
            "future prediction length (k_var steps in the future)": self.k_value,
            "start_time": datetime.now().strftime("%d-%m-%Y_%H-%M-%S"),
            "my_notes": wdb_notes
            }
        )
        self.wandb_run_url = wandb.run.url
    
    '''
    Train the model
        t_value: the t in L1, L2, L3 formular
        Loss1: L1 loss: from i= 1 to t_value
        Loss2: L2 loss: from i= t_value + 1 to t_value + k_value - 1
        Loss3: L3 loss: from i= 0 to t_value-1
    '''
    def train(self, data_train:tuple, data_val:tuple, checkpoint_path=None):
        start_t = time.time()
        # check if data is proper
        if not self.utils.proper_data_check(data_train, data_val, self.t_value, self.k_value):
            # print in red color
            print('\033[91m' + 'Data is not proper. Please check again!' + '\033[0m')
            return

        if checkpoint_path:
            start_epoch = self.load_checkpoint(checkpoint_path) + 1
        else:
            start_epoch = 0

        # 1. ----- Prepare data -----
        dl_train, dl_val = self.utils.prepare_data_loaders(data_train, data_val, self.batch_size_train, self.batch_size_val)

        # 2. ----- Training -----
        plot_data_loss1 = []
        plot_data_loss2 = []
        plot_data_loss3 = []
        plot_data_loss_total = []
        for epoch in range(start_epoch, self.num_epochs):
            self.lstm.train()
            self.encoder.train()
            self.decoder.train()
            loss_total_train_log = 0.0
            loss_1_train_log = 0.0
            loss_2_train_log = 0.0
            loss_3_train_log = 0.0

            for batch_x_train, batch_y_train in dl_train:
                # ----- 2.1 Setup input and labels -----
                batch_x_train, batch_y_train = batch_x_train.to(self.device), batch_y_train.to(self.device)
                inputs = batch_x_train
                # input('TODO: check inputs shape: ')   # t steps
                # print('inputs shape: ', batch_x_train.shape)
                # print('label entire shape: ', batch_y_train.shape)

                l1_label_train = batch_y_train[:, 1:self.t_value+1, :]     # t steps
                l2_label_train = batch_y_train[:, self.t_value+1:, :]       # k-1 steps
                l3_label_train = batch_y_train[:, :self.t_value, :]         # t steps
                # input('TODO: check labels of l1, l2, l3')
                # print('l1_label_train shape: ', l1_label_train.shape)
                # print('l2_label_train shape: ', l2_label_train.shape)
                # print('l3_label_train shape: ', l3_label_train.shape)

                self.optimizer.zero_grad() # should be before model forward

                # ----- 2.2 Training -----
                hi = torch.zeros(self.num_layers, inputs.size(0), self.hidden_size).to(inputs.device)
                ci = torch.zeros(self.num_layers, inputs.size(0), self.hidden_size).to(inputs.device)
                inputs_e = self.encoder(inputs)     # t steps
                out_seq_lstm = []

                for i in range(self.t_value + self.k_value -1): # t + k - 1 steps
                    if i < self.t_value:                        # t first steps from given data
                        lstm_input = inputs_e[:, i:i+1, :]
                    else:                                       # k - 1 steps from predicted data
                        lstm_input = output[:, -1:, :]
                    
                    # get prediction and update hidden states hi, ci
                    output, hi, ci = self.lstm(lstm_input, hi, ci)
                    out_seq_lstm.append(output)

                out_seq_lstm = torch.cat(out_seq_lstm, dim=1)
                predicted_seq = self.decoder(out_seq_lstm)

                # ----- 2.3 Calculate loss -----
                l1_pred_train = predicted_seq[:, :self.t_value, :]  # t steps
                l2_pred_train = predicted_seq[:, self.t_value:, :]  # k-1 steps

                # input('check l1_pred_train, l2_pred_train shape')
                # print('l1_pred_train shape: ', l1_pred_train.shape)
                # print('l2_pred_train shape: ', l2_pred_train.shape)
                
                loss_1_train = self.criterion(l1_label_train, l1_pred_train) / l1_label_train.shape[1]              # /(self.t_value)
                loss_2_train = self.criterion(l2_label_train, l2_pred_train) / l2_label_train.shape[1]              # /(self.k_value-1)
                loss_3_train = self.criterion(l3_label_train, self.decoder(inputs_e)) / l3_label_train.shape[1]     # /(self.t_value)
                training_loss_total = loss_1_train + loss_2_train + loss_3_train

                # ----- 2.4 Backward pass -----
                training_loss_total.backward()
                self.optimizer.step()
            

                # log loss
                loss_total_train_log += training_loss_total.item()
                loss_1_train_log += loss_1_train.item() * batch_x_train.size(0)
                loss_2_train_log += loss_2_train.item() * batch_x_train.size(0)
                loss_3_train_log += loss_3_train.item() * batch_x_train.size(0)


            # get average value of loss
            loss_1_train_log/=len(dl_train.dataset)
            loss_2_train_log/=len(dl_train.dataset)
            loss_3_train_log/=len(dl_train.dataset)
            loss_total_train_log/=len(dl_train.dataset)

            plot_data_loss1.append(loss_1_train_log)
            plot_data_loss2.append(loss_2_train_log)
            plot_data_loss3.append(loss_3_train_log)
            plot_data_loss_total.append(loss_total_train_log)
            loss_all_data = [plot_data_loss1, plot_data_loss2, plot_data_loss3, plot_data_loss_total]
            if epoch % self.save_interval == 0:
                self.save_model(epoch, len(data_train), start_t, loss_all_data)

            traing_time = time.time() - start_t

            # 2. ----- FOR VALIDATION -----
            loss_total_val_log, mean_mse_all, mean_mse_xyz, \
            mean_ade, mean_nade, mean_final_step_err, \
            mean_nade_future, capture_success_rate = self.validate_and_score(dl_val)
            validate_time = time.time() - start_t - traing_time
      
            # 3. ----- FOR WANDB LOG -----
            wandb.log({
                "training_loss1": loss_1_train_log,
                "training_loss2": loss_2_train_log,
                "training_loss3": loss_3_train_log,
                "training_loss_total": loss_total_train_log,
                "valid_loss_total": loss_total_val_log,
                "scored_mean_mse_all": mean_mse_all,
                "scored_mean_mse_xyz": mean_mse_xyz,
                "scored_mean_ade": mean_ade,
                "scored_mean_nade": mean_nade,
                "scored_mean_final_dist": mean_final_step_err,
                "scored_mean_nade_future": mean_nade_future,
                "capture_success_rate": capture_success_rate,
                },
                step=epoch
            )

            if (epoch) % 10 == 0:
                print('\n-----------------------------------')
                print(f'Epoch [{epoch}/{self.num_epochs}], Loss: {loss_total_train_log:.6f}, traing time: {traing_time:.2f} s ({traing_time/(traing_time+validate_time)*100} %), validate time: {validate_time:.2f} s ({validate_time/(traing_time+validate_time)*100} %)')
            
        final_model_dir = self.save_model(epoch, len(data_train), start_t, loss_all_data)
        wandb.finish()
        return final_model_dir

        
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.lstm.load_state_dict(checkpoint['model_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        return epoch
    
    def save_model(self, epoch, data_num, start_t, loss_all_data):    
        epoch_num_midway = epoch
        data_num = data_num

        training_t = time.time() - start_t
        training_t = round(training_t/60, 2)
        train_time = str(training_t).replace('.', '-')
        print('epoch_num: ', self.num_epochs)
        print('data_num: ', data_num)
        print('seq_length: ', self.t_value)
        print('prediction_steps: ', self.k_value)
        print(f'Training time: {(time.time() - start_t)/60:.2f} mins')
        # calculate training time left
        training_time_left = (time.time() - start_t) * (self.num_epochs - (epoch+1)) / (epoch+1)
        print(f'Training time left: {training_time_left/60/60:.2f} hours\n')

        sub_folder = ('epochs' + str(epoch_num_midway) 
                    + '_data' + str(data_num) 
                    + '_batchsize' + str(self.batch_size_train) 
                    + '_hiddensize' + str(self.hidden_size) 
                    + '_seq' + str(self.t_value) 
                    + '_pred' + str(self.k_value)
                    + '_timemin' + str(train_time))

        model_dir = os.path.join(self.model_dir, sub_folder)
        # Save the model
        # Create dir if not exist
        os.makedirs(model_dir, exist_ok=True)
        encoder_model_path = os.path.join(model_dir, 'encoder_model.pth') 
        torch.save(self.encoder.state_dict(), encoder_model_path)

        lstm_model_path = os.path.join(model_dir, 'lstm_model.pth') 
        torch.save(self.lstm.state_dict(), lstm_model_path)

        decoder_model_path = os.path.join(model_dir, 'decoder_model.pth') 
        torch.save(self.decoder.state_dict(), decoder_model_path)

        # Save midways
        midway_checkpoint_dir = os.path.join(model_dir, 'midway_checkpoint')
        os.makedirs(midway_checkpoint_dir, exist_ok=True)
        # Save optimizer state
        optimizer_state_path = os.path.join(midway_checkpoint_dir, 'optimizer_state.pth')
        torch.save(self.optimizer.state_dict(), optimizer_state_path)

        checkpoint_path = os.path.join(midway_checkpoint_dir, 'checkpoint.pth')
        torch.save({
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'model_state_dict': self.lstm.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss_all_data[-1][-1],
        }, checkpoint_path)
        
        
        loss_graph_image_path = self.utils.save_loss(loss_all_data, model_dir)
        
        self.utils.save_model_info(self.data_dir, model_dir, data_num, self.num_epochs, self.batch_size_train, self.t_value, self.k_value, start_t, training_t, self.wandb_run_url, loss_all_data, loss_graph_image_path)
        print(f'Models were saved to {model_dir}')
        return model_dir
   
    def load_model(self, model_weights_dir):
        encoder_model_path = self.utils.look_for_file(model_weights_dir, 'encoder_model.pth')
        lstm_model_path = self.utils.look_for_file(model_weights_dir, 'lstm_model.pth')
        decoder_model_path = self.utils.look_for_file(model_weights_dir, 'decoder_model.pth')

        self.encoder.load_state_dict(torch.load(encoder_model_path))
        self.encoder.to(self.device)

        self.lstm.load_state_dict(torch.load(lstm_model_path))
        self.lstm.to(self.device)

        self.decoder.load_state_dict(torch.load(decoder_model_path))
        self.decoder.to(self.device)
    
    def predict(self, inputs, evaluation=False):
        self.lstm.eval()
        self.encoder.eval()
        self.decoder.eval()

        if evaluation:
            inputs = torch.tensor(inputs, dtype = torch.float32).to(self.device)

        with torch.no_grad():
            hi = torch.zeros(self.num_layers, inputs.size(0), self.hidden_size).to(inputs.device)
            ci = torch.zeros(self.num_layers, inputs.size(0), self.hidden_size).to(inputs.device)
            inputs_e = self.encoder(inputs)     # t steps
            out_seq_lstm = []

            for i in range(self.t_value + self.k_value -1): # t + k - 1 steps
                if i < self.t_value:                        # t first steps from given data
                    lstm_input = inputs_e[:, i:i+1, :]
                else:                                       # k - 1 steps from predicted data
                    lstm_input = output[:, -1:, :]
                
                # get prediction and update hidden states hi, ci
                output, hi, ci = self.lstm(lstm_input, hi, ci)
                out_seq_lstm.append(output)

            out_seq_lstm = torch.cat(out_seq_lstm, dim=1)
            predicted_seq = self.decoder(out_seq_lstm)
            return predicted_seq
        
    def validate_and_score(self, dl_val):
        self.lstm.eval()
        self.encoder.eval()
        self.decoder.eval()

        loss_total_val_log = 0.0
        sum_mse_all = 0.0
        sum_mse_xyz = 0.0
        sum_ade = 0.0
        sum_nade = 0.0
        sum_final_dist = 0.0
        sum_nade_future = 0.0
        sime_capture_success_times = 0

        with torch.no_grad():
            for batch_x_val, batch_y_val in dl_val:
                # ----- 1. Setup input and labels -----
                batch_x_val, batch_y_val = batch_x_val.to(self.device), batch_y_val.to(self.device)
                inputs = batch_x_val
                l1_label_val = batch_y_val[:, 1:self.t_value+1, :]      # t steps
                l2_label_val = batch_y_val[:, self.t_value+1:, :]       # k-1 steps
                l3_label_val = batch_y_val[:, :self.t_value, :]         # t steps
                label_seq = batch_y_val[:, 1:, :]                       # t + k - 1 steps from 1 to t + k -1
                # ----- 2. Predict -----
                predicted_seq = self.predict(inputs)

                # ----- 3. Calculate loss and scores -----
                l1_pred_val = predicted_seq[:, :self.t_value, :]     # t steps
                l2_pred_val = predicted_seq[:, self.t_value:, :]     # k-1 steps


                # print('\nl1_label_val shape: ', l1_label_val.shape)
                # print('l1_pred_val shape: ', l1_pred_val.shape)
                # input('TODO: check l1_label_val, l1_pred_val shape')

                loss_1_val = self.criterion(l1_label_val, l1_pred_val) / l1_label_val.shape[1]                          # /(self.t_value)
                loss_2_val = self.criterion(l2_label_val, l2_pred_val) / l2_label_val.shape[1]                          # /(self.k_value-1)
                loss_3_val = self.criterion(l3_label_val, self.decoder(self.encoder(inputs))) / l3_label_val.shape[1]   # /(self.t_value)
                loss_total_val = loss_1_val + loss_2_val + loss_3_val

                loss_total_val_log += loss_total_val.item()
                
                # print('\npredicted_seq shape: ', predicted_seq.shape)
                # print('label shape: ', label_seq.shape)
                # input('TODO: check predicted_seq, label shape')

                batch_mean_mse_all, batch_mean_mse_xyz, batch_mean_ade, \
                (batch_mean_nade, batch_var_nade), (batch_mean_fe, batch_var_fe), \
                batch_mean_nade_future, capture_success = self.utils.score_all_predictions(predicted_seq.cpu().numpy(), 
                                                                                           label_seq.cpu().numpy(), 
                                                                                           self.k_value,
                                                                                           capture_thres=0.1)
                # because MSE divides by batch_size, but we need to sum all to get the total loss and calculate the mean value ourselves, 
                # so in each batch we need to multiply by batch_size
                current_batch_size = batch_x_val.size(0)
                sum_mse_all += batch_mean_mse_all*current_batch_size
                sum_mse_xyz += batch_mean_mse_xyz*current_batch_size
                sum_ade += batch_mean_ade*current_batch_size
                sum_nade += batch_mean_nade*current_batch_size
                sum_final_dist += batch_mean_fe*current_batch_size
                sum_nade_future += batch_mean_nade_future*current_batch_size
                sime_capture_success_times += capture_success*current_batch_size/100
        
        # get mean value of scored data
        mean_mse_all = sum_mse_all/len(dl_val.dataset)
        mean_mse_xyz = sum_mse_xyz/len(dl_val.dataset)
        mean_ade = sum_ade/len(dl_val.dataset)
        mean_nade = sum_nade/len(dl_val.dataset)
        mean_final_step_err = sum_final_dist/len(dl_val.dataset)
        mean_nade_future = sum_nade_future/len(dl_val.dataset)
        capture_success_rate = sime_capture_success_times/len(dl_val.dataset)*100

        # get average value of loss
        loss_total_val_log/=len(dl_val.dataset)
        return loss_total_val_log, mean_mse_all, mean_mse_xyz, mean_ade, mean_nade, mean_final_step_err, mean_nade_future, capture_success_rate

def main():
    # ===================== TRAINING =====================
    # set seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    thrown_object = 'big_plane'
    data_folder = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_prediction_ws/src/nae/data/new_dataset_no_orientation/big_plane/min_len_65/new_data_format/big_plane'

    # Learning parameters
    step_begin_dynamic_nae = 1
    step_increment_dynamic_nae = 1

    future_pred_len = 35
    training_params = {
        'input_len': step_begin_dynamic_nae,
        'future_pred_len': step_increment_dynamic_nae,
        'num_epochs': 10000,
        'batch_size_train': 512,    
        'batch_size_val': 128,
        'save_interval': 10,
        'thrown_object' : thrown_object + '-input-' + str(step_begin_dynamic_nae) + 'pred-' + str(step_increment_dynamic_nae)
    }
    # Wandb parameters
    wdb_run_id=None   # 't5nlloi0'
    wdb_resume=None   # 'allow'
    model_params = {
        'input_size': 9,
        'hidden_size': 128,
        'output_size': 9,
        'num_layers_lstm': 2,
        'lr': 0.0002
    }

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
    input_label_dynamic_enerator = InputLabelDynamicGenerator()
    data_train  = input_label_dynamic_enerator.create_dynamic_input_label_pairs(data_train, 
                                                                                step_begin=step_begin_dynamic_nae, 
                                                                                increment=step_increment_dynamic_nae)
    data_val    = input_label_dynamic_enerator.create_dynamic_input_label_pairs(data_val, 
                                                                                step_begin=step_begin_dynamic_nae, 
                                                                                increment=step_increment_dynamic_nae)
    data_test   = input_label_dynamic_enerator.create_dynamic_input_label_pairs(data_test,
                                                                                step_begin=step_begin_dynamic_nae, 
                                                                                increment=step_increment_dynamic_nae)
    
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
    nae = NAE(**model_params, **training_params, data_dir=data_folder, device=DEVICE)
    nae.init_wandb(project_name='nae',
                   run_id=wdb_run_id, 
                   resume=wdb_resume,
                   wdb_notes=wdb_notes)

    checkpoint_path = None
    saved_model_dir = nae.train(data_train, data_val, checkpoint_path=checkpoint_path)

if __name__ == '__main__':
    main()