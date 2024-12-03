import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as TorchDataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch.optim as optim
import random
import wandb
from datetime import datetime
import os
import time

from python_utils.printer import Printer
from utils.utils import NAE_Utils
from utils.submodules.training_utils.data_loader import DataLoader as NAEDataLoader
from utils.submodules.preprocess_utils.data_raw_correction_checker import RoCatRLDataRawCorrectionChecker
from utils.submodules.training_utils.input_label_generator import InputLabelGenerator

class TimeSeriesDataset(Dataset):
    def __init__(self, num_samples=100, max_len=10, feature_size=32):
        self.data = []
        for _ in range(num_samples):
            seq_len = random.randint(3, max_len)
            seq_len_in = random.randint(1, seq_len-1)

            seq = torch.rand(seq_len, feature_size)  # 1 feature per timestep
            input_seq = seq[:seq_len_in]
            label_teafo_seq = seq[1:seq_len_in+1]
            label_aureg_seq = seq[seq_len_in+1 : ]
            label_reconstruction_seq = seq[:seq_len_in]

            self.data.append((input_seq, label_teafo_seq, label_aureg_seq, label_reconstruction_seq))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.fc(x)
        out = self.tanh(out)
        return out
    
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc(x)
        return out

class VLSLSTM(nn.Module):
    '''
    Variable-Length Sequence LSTM class
    '''
    def __init__(self, input_size, hidden_size, num_layers):
        super(VLSLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x, lengths_teacher_forcing, lengths_autoregressive):
        this_batch_size = x.size(0)
        packed_x = pack_padded_sequence(x, lengths_teacher_forcing, batch_first=True, enforce_sorted=False)


        hi = torch.zeros(self.num_layers, this_batch_size, self.hidden_size).to(x.device)
        ci = torch.zeros(self.num_layers, this_batch_size, self.hidden_size).to(x.device)
        outputs_teafo_unpad = [] # teacher forcing output sequences

        #----- teacher forcing -----
        outputs, (hi, ci) = self.lstm(packed_x, (hi, ci))

        # unpack and only keep real steps, not the padding based on lengths_teacher_forcing
        # Unpack dữ liệu packed outputs
        outputs_teafo_pad, len_real_teafo = pad_packed_sequence(outputs, batch_first=True)
        # print('lengths_teacher_forcing: ', lengths_teacher_forcing)
        # print('len_real_teafo: ', len_real_teafo)


        # Slicing để lấy từng chuỗi theo độ dài thực
        batch_indices = torch.arange(outputs_teafo_pad.size(0))  # [0, 1, 2]
        outputs_teafo_unpad = [outputs_teafo_pad[i, :len_real_teafo[i], :] for i in batch_indices]

        
        #----- autoregressive -----
        # 1. Lấy output cuối cùng của teacher forcing
        input_aureg_init = [out[-1] for out in outputs_teafo_unpad]
        input_aureg_init = torch.stack(input_aureg_init, dim=0).unsqueeze(1)  # (batch_size, 1, hidden_size)
        # print('check input_aureg_init shape: ', input_aureg_init.shape)
        # print('check hi shape: ', hi.shape)
        # print('check ci shape: ', ci.shape)
        # input()

        outputs_aureg_unpad = self.auto_regressive_loop(input_aureg_init, hi, ci, lengths_autoregressive)
        # Nối 2 chuỗi output lại với nhau (dim=1 là chiều thời gian)

        # print('outputs_aureg_unpad length: ', len(outputs_aureg_unpad))
        # print('lengths_teacher_forcing: ', lengths_teacher_forcing)
        # print('lengths_autoregressive: ', lengths_autoregressive)
        # for seq in outputs_aureg_unpad:
        #     print('     seq shape: ', seq.shape)

        # out_seq = self.concat_output_seq(outputs_teafo_unpad, outputs_aureg_unpad)

        # padding outputs_aureg_unpad
        outputs_aureg_pad = pad_sequence(outputs_aureg_unpad, batch_first=True) 
        # print('check output_teafo_pad shape: ', output_teafo_pad.shape)
        # print('check outputs_aureg_pad shape: ', outputs_aureg_pad.shape)
        # input()
        return outputs_teafo_pad, outputs_aureg_pad
    
    # def concat_output_seq(self, outputs_teafo_unpad, out_seq_aureg):
    #     # Nối 2 chuỗi output lại với nhau (dim=1 là chiều thời gian)
    #     out_seqs = []
    #     print('check outputs_teafo_unpad shape: ', len(outputs_teafo_unpad))
    #     print('check out_seq_aureg shape: ', len(out_seq_aureg))
    #     for seq_teafo, seq_aureg in zip(outputs_teafo_unpad, out_seq_aureg):
    #         # extend the teacher forcing sequence with the autoregressive sequence (tensors)
    #         # seq_teafo = torch.cat([seq_teafo, seq_aureg], dim=1)
    #         print('check seq_teafo shape: ', seq_teafo.shape)
    #         print('check seq_aureg shape: ', seq_aureg.shape)
    #         # merge two sequences
    #         seq_teafo = torch.cat([seq_teafo, seq_aureg], dim=0)
    #         print('len seq_teafo: ', seq_teafo.shape)
    #         input()
    #         out_seqs.append(seq_teafo)
    #     return out_seqs
        
    def auto_regressive_loop(self, input_aureg_init, hi, ci, lengths):
        """
        Thực hiện dự đoán autoregressive.
        
        Args:
            batch_x_padded: Tensor đầu vào đã được padding, kích thước (batch_size, seq_len, feature_size).
            lengths: Độ dài thực của từng chuỗi trong batch, kích thước (batch_size,).
        
        Returns:
            output_seq: Tensor đầu ra của mô hình, kích thước (batch_size, seq_len, hidden_size).
        """
        batch_size = input_aureg_init.size(0)
        max_len = lengths.max().item()
        hidden_size = self.lstm.hidden_size

        # Tạo mask cho từng bước
        mask = torch.arange(max_len).unsqueeze(0).to(lengths.device) < lengths.unsqueeze(1)

        output_seq = torch.zeros(batch_size, max_len, hidden_size).to(input_aureg_init[0].device)


        # Lấy bước đầu tiên làm đầu vào ban đầu
        lstm_input = input_aureg_init  # Kích thước (batch_size, 1, feature_size)

        # Duyệt qua từng bước thời gian
        for t in range(max_len):
            current_mask = mask[:, t]  # Mask tại bước thời gian t
            if not current_mask.any():  # Nếu tất cả đều là padding, dừng lại
                break

            # Lọc các chuỗi thực tại bước thời gian hiện tại
            lstm_input = lstm_input[current_mask]  # (num_real_sequences, 1, feature_size)
            hi_current = hi[:, current_mask, :]  # (num_layers, num_real_sequences, hidden_size)
            ci_current = ci[:, current_mask, :]  # (num_layers, num_real_sequences, hidden_size)

            # Truyền qua LSTM
            output, (hi_new, ci_new) = self.lstm(lstm_input, (hi_current, ci_current))

            # Cập nhật trạng thái ẩn cho các chuỗi thực
            hi[:, current_mask, :] = hi_new
            ci[:, current_mask, :] = ci_new

            # Lưu output vào tensor output_seq
            temp_output = torch.zeros(batch_size, 1, hidden_size).to(output.device)

            temp_output[current_mask] = output  # Ghi output của các chuỗi thực

            # Cập nhật đầu vào cho bước tiếp theo
            lstm_input = temp_output[:, -1:, :]  # Sử dụng output hiện tại làm input tiếp theo

            # output_seq.append(temp_output[current_mask])  # Thêm vào danh sách output
            output_seq[:, t:t+1, :] = temp_output
        
        # print('check output_seq shape: ', output_seq.shape)
        # filter padding and save to a list
        output_seq_final = [output_seq[i, :lengths[i], :] for i in range(batch_size)]

        return output_seq_final

    

#------------------------- TRAINING -------------------------
class NAEDynamicLSTM():
    def __init__(self, input_size, hidden_size, output_size, num_layers_lstm, lr, 
                 num_epochs, batch_size_train, batch_size_val, save_interval, thrown_object,
                 device=torch.device('cuda'),
                 data_dir=''):
        self.utils = NAE_Utils()
        self.device = device

        # model architecture params
        self.hidden_size = hidden_size
        self.num_layers = num_layers_lstm
        self.lr = lr

        # training params
        self.num_epochs = num_epochs
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.save_interval = save_interval
        self.thrown_object = thrown_object + '_model'
        self.data_dir = data_dir

        self.run_name = f"NAE_DYNAMIC-model_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}_hiddensize{hidden_size}"
        self.model_dir = os.path.join('models', self.thrown_object, self.run_name)

        # Initialize model, loss, optimizer
        self.encoder = Encoder(input_size, hidden_size).to(device)
        self.vls_lstm = VLSLSTM(hidden_size, hidden_size, num_layers_lstm).to(device)
        self.decoder = Decoder(hidden_size, output_size).to(device)

        self.criterion = nn.MSELoss(reduction='none').to(device)  # NOTE: Reduction 'none' to apply masking, default is 'mean'
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.vls_lstm.parameters()) + list(self.decoder.parameters()), lr=lr)

        self.util_printer = Printer()

        print('\n-----------------------------------')
        print('Parameters number of encoder: ', self.utils.count_parameters(self.encoder))
        print('Parameters number of LSTM: ', self.utils.count_parameters(self.vls_lstm))
        print('Parameters number of decoder: ', self.utils.count_parameters(self.decoder))
        print('Total number of parameters: ', self.utils.count_parameters(self.encoder) + self.utils.count_parameters(self.vls_lstm) + self.utils.count_parameters(self.decoder))


    def collate_pad_fn(self, batch):
        inputs, labels_teafo, labels_aureg, label_reconstruction = zip(*batch)

        lengths_in = [len(seq) for seq in inputs]
        lengths_teafo = [len(seq) for seq in labels_teafo]
        lengths_aureg = [len(seq) for seq in labels_aureg]
        lengths_reconstruction = [len(seq) for seq in label_reconstruction]

        # Check if data is proper
        for lin, lte, lre in zip(lengths_in, lengths_teafo, lengths_reconstruction):
            # check if lin, lte, lre are the same length
            assert lin == lte == lre, f'Lengths are not the same: {lin}, {lte}, {lre}'
        
        
        # Padding sequences to have same length in batch
        inputs_pad = pad_sequence(inputs, batch_first=True)  # Shape: (batch_size, max_seq_len_in, 1)
        # labels_pad = pad_sequence(labels, batch_first=True)  # Shape: (batch_size, max_seq_len_out, 1)
        labels_teafo_pad = pad_sequence(labels_teafo, batch_first=True)
        labels_aureg_pad = pad_sequence(labels_aureg, batch_first=True)
        labels_reconstruction_pad = pad_sequence(label_reconstruction, batch_first=True)
        
        return inputs_pad, \
            (labels_teafo_pad, lengths_teafo), \
            (labels_aureg_pad, lengths_aureg), \
            (labels_reconstruction_pad, lengths_reconstruction)

    def train(self, data_train, data_val, checkpoint_path=None):
        start_t = time.time()

        dataloader_train = TorchDataLoader(data_train, batch_size=self.batch_size_train, collate_fn=lambda x: self.collate_pad_fn(x), shuffle=True)

        for epoch in range(self.num_epochs):
            self.encoder.train()
            self.vls_lstm.train()
            self.decoder.train()

            # loss_epoch_log = 0
            loss_total_train_log = 0.0
            loss_1_train_log = 0.0
            loss_2_train_log = 0.0
            loss_3_train_log = 0.0
            loss_all_log = []
            for batch in dataloader_train:

                inputs, \
                (labels_teafo_pad, lengths_teafo), \
                (labels_aureg_pad, lengths_aureg), \
                (labels_reconstruction_pad, lengths_reconstruction) = batch

                inputs = inputs.to(self.device)
                labels_teafo_pad = labels_teafo_pad.to(self.device)
                labels_aureg_pad = labels_aureg_pad.to(self.device)
                labels_reconstruction_pad= labels_reconstruction_pad.to(self.device)

                lengths_teafo= torch.tensor(lengths_teafo, dtype=torch.int64)
                lengths_aureg= torch.tensor(lengths_aureg, dtype=torch.int64)
                lengths_reconstruction= torch.tensor(lengths_reconstruction, dtype=torch.int64)

                inputs_lstm = self.encoder(inputs.float())
                outputs_teafo_pad, output_aureg_pad = self.vls_lstm(inputs_lstm, lengths_teafo, lengths_aureg)

                output_teafo_pad_de = self.decoder(outputs_teafo_pad)
                output_aureg_pad_de = self.decoder(output_aureg_pad)

                ##  ----- LOSS 1: TEACHER FORCING -----
                loss_1 = self.criterion(output_teafo_pad_de, labels_teafo_pad).sum(dim=-1)  # Shape: (batch_size, max_seq_len_out)
                # Tạo mask dựa trên chiều dài thực
                mask_teafo = torch.arange(max(lengths_teafo)).expand(len(lengths_teafo), max(lengths_teafo)) < lengths_teafo.unsqueeze(1) # shape: (batch_size, max_seq_len_out)
                mask_teafo = mask_teafo.to(loss_1.device)
                loss_1_mask = loss_1 * mask_teafo  # Masked loss
                loss_1_mean = 0
                if mask_teafo.sum() != 0:
                    loss_1_mean = loss_1_mask.sum() / mask_teafo.sum()

                ## ----- LOSS 2: AUTOREGRESSIVE -----
                loss_2 = self.criterion(output_aureg_pad_de, labels_aureg_pad).sum(dim=-1)
                mask_aureg = torch.arange(max(lengths_aureg)).expand(len(lengths_aureg), max(lengths_aureg)) < lengths_aureg.unsqueeze(1) # shape: (batch_size, max_seq_len_out)
                mask_aureg = mask_aureg.to(loss_2.device)
                loss_2_mask = loss_2 * mask_aureg
                loss_2_mean = 0
                if mask_aureg.sum() != 0:
                    loss_2_mean = loss_2_mask.sum() / mask_aureg.sum()

                ## ----- LOSS 3: RECONSTRUCTION -----
                loss_3 = self.criterion(self.decoder(inputs_lstm), labels_reconstruction_pad).sum(dim=-1)
                mask_reconstruction = torch.arange(max(lengths_reconstruction)).expand(len(lengths_reconstruction), max(lengths_reconstruction)) < lengths_reconstruction.unsqueeze(1) # shape: (batch_size, max_seq_len_out)
                mask_reconstruction = mask_reconstruction.to(loss_3.device)
                loss_3_mask = loss_3 * mask_reconstruction
                loss_3_mean = 0
                if mask_reconstruction.sum() != 0:
                    loss_3_mean = loss_3_mask.sum() / mask_reconstruction.sum()

                loss_mean = loss_1_mean + loss_2_mean + loss_3_mean

                # Backward pass
                self.optimizer.zero_grad()
                loss_mean.backward()
                self.optimizer.step()

                if torch.isnan(loss_mean):
                    self.util_printer.print_red("Loss contains NaN!", background=True)
                    input()

                loss_total_train_log += loss_mean.item()
                loss_1_train_log += loss_1_mean.item()
                loss_2_train_log += loss_2_mean.item()
                loss_3_train_log += loss_3_mean.item()
            
            # get average loss over all batches
            loss_total_train_log /= len(dataloader_train)
            loss_1_train_log /= len(dataloader_train)
            loss_2_train_log /= len(dataloader_train)
            loss_3_train_log /= len(dataloader_train)
            loss_all_log.append([loss_1_train_log, loss_2_train_log, loss_3_train_log, loss_total_train_log])

            if epoch % self.save_interval == 0:
                self.save_model(epoch, len(data_train), start_t, loss_all_log)
            traing_time = time.time() - start_t
            


            # 2. ----- FOR VALIDATION -----
            dataloader_val = TorchDataLoader(data_val, batch_size=self.batch_size_val, collate_fn=lambda x: self.collate_pad_fn(x), shuffle=True)
            mean_loss_total_val_log, \
            mean_ade_entire, mean_ade_future, \
            mean_nade_entire, mean_nade_future, \
            mean_final_step_err, capture_success_rate = self.validate_and_score(dataloader_val)
            validate_time = time.time() - start_t - traing_time
      
            # 3. ----- FOR WANDB LOG -----
            wandb.log({
                "training_loss1": loss_1_train_log,
                "training_loss2": loss_2_train_log,
                "training_loss3": loss_3_train_log,
                "training_loss_total": loss_total_train_log,
                "valid_loss_total": mean_loss_total_val_log,
                "valid_mean_ade_entire": mean_ade_entire,
                "valid_mean_ade_future": mean_ade_future,
                "valid_mean_nade_entire": mean_nade_entire,
                "valid_mean_nade_future": mean_nade_future,
                "valid_mean_final_step_err": mean_final_step_err,
                "valid_capture_success_rate": capture_success_rate,
                "training_time_mins": traing_time/60
                },
                step=epoch
            )

            if (epoch) % 10 == 0:
                self.util_printer.print_green(f'Epoch [{epoch}/{self.num_epochs}], Loss: {loss_total_train_log:.6f}, traing time: {traing_time:.2f} s ({traing_time/(traing_time+validate_time)*100} %), validate time: {validate_time:.2f} s ({validate_time/(traing_time+validate_time)*100} %)')
                print('\n-----------------------------------')

        final_model_dir = self.save_model(epoch, len(data_train), start_t, loss_all_log)

        wandb.finish()
        return final_model_dir
    
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
            "start_time": datetime.now().strftime("%d-%m-%Y_%H-%M-%S"),
            "my_notes": wdb_notes
            }
        )
        self.wandb_run_url = wandb.run.url
    
    def save_model(self, epoch, data_num, start_t, loss_all_data):    
        epoch_num_midway = epoch
        data_num = data_num

        training_t = time.time() - start_t
        training_t = round(training_t/60, 2)
        train_time = str(training_t).replace('.', '-')
        print('epoch_num: ', self.num_epochs)
        print('data_num: ', data_num)
        print(f'Training time: {(time.time() - start_t)/60:.2f} mins')
        # calculate training time left
        training_time_left = (time.time() - start_t) * (self.num_epochs - (epoch+1)) / (epoch+1)
        print(f'Training time left: {training_time_left/60/60:.2f} hours\n')

        sub_folder = ('epochs' + str(epoch_num_midway) 
                    + '_data' + str(data_num) 
                    + '_batchsize' + str(self.batch_size_train) 
                    + '_hiddensize' + str(self.hidden_size) 
                    + '_timemin' + str(train_time)
                    + '_NAE_DYNAMIC')

        model_dir = os.path.join(self.model_dir, sub_folder)
        # Save the model
        # Create dir if not exist
        os.makedirs(model_dir, exist_ok=True)
        encoder_model_path = os.path.join(model_dir, 'encoder_model.pth') 
        torch.save(self.encoder.state_dict(), encoder_model_path)

        lstm_model_path = os.path.join(model_dir, 'lstm_model.pth') 
        torch.save(self.vls_lstm.state_dict(), lstm_model_path)

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
            'model_state_dict': self.vls_lstm.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss_all_data,
        }, checkpoint_path)
        
        
        # loss_graph_image_path = self.utils.save_loss(loss_all_data, model_dir)
        
        self.utils.save_model_info(self.data_dir, model_dir, data_num, self.num_epochs, self.batch_size_train, start_t, training_t, self.wandb_run_url, loss_all_data)
        print(f'Models were saved to {model_dir}')
        return model_dir

    def validate_and_score(self, dl_val):
        self.vls_lstm.eval()
        self.encoder.eval()
        self.decoder.eval()

        loss_total_val_log = 0.0
        # sum_mse_all = 0.0
        # sum_mse_xyz = 0.0
        sum_ade_entire = 0.0
        sum_ade_future = 0.0
        sum_nade_entire = 0.0
        sum_nade_future = 0.0
        sum_final_step_err = 0.0
        sum_capture_success_rate = 0.0

        with torch.no_grad():
            for batch in dl_val:
                inputs, \
                (labels_teafo_pad, lengths_teafo), \
                (labels_aureg_pad, lengths_aureg), \
                (labels_reconstruction_pad, lengths_reconstruction) = batch

                inputs = inputs.to(self.device)
                labels_teafo_pad = labels_teafo_pad.to(self.device)
                labels_aureg_pad = labels_aureg_pad.to(self.device)
                labels_reconstruction_pad= labels_reconstruction_pad.to(self.device)

                lengths_teafo= torch.tensor(lengths_teafo, dtype=torch.int64)
                lengths_aureg= torch.tensor(lengths_aureg, dtype=torch.int64)
                lengths_reconstruction= torch.tensor(lengths_reconstruction, dtype=torch.int64)

                inputs_lstm = self.encoder(inputs.float())
                outputs_teafo_pad, output_aureg_pad = self.vls_lstm(inputs_lstm, lengths_teafo, lengths_aureg)

                output_teafo_pad_de = self.decoder(outputs_teafo_pad)
                output_aureg_pad_de = self.decoder(output_aureg_pad)

                ##  ----- LOSS 1: TEACHER FORCING -----
                loss_1 = self.criterion(output_teafo_pad_de, labels_teafo_pad).sum(dim=-1)  # Shape: (batch_size, max_seq_len_out)
                # Tạo mask dựa trên chiều dài thực
                mask_teafo = torch.arange(max(lengths_teafo)).expand(len(lengths_teafo), max(lengths_teafo)) < lengths_teafo.unsqueeze(1) # shape: (batch_size, max_seq_len_out)
                mask_teafo = mask_teafo.to(loss_1.device)
                loss_1_mask = loss_1 * mask_teafo  # Masked loss
                loss_1_mean = 0
                if mask_teafo.sum() != 0:
                    loss_1_mean = loss_1_mask.sum() / mask_teafo.sum()

                ## ----- LOSS 2: AUTOREGRESSIVE -----
                loss_2 = self.criterion(output_aureg_pad_de, labels_aureg_pad).sum(dim=-1)
                mask_aureg = torch.arange(max(lengths_aureg)).expand(len(lengths_aureg), max(lengths_aureg)) < lengths_aureg.unsqueeze(1) # shape: (batch_size, max_seq_len_out)
                mask_aureg = mask_aureg.to(loss_2.device)
                loss_2_mask = loss_2 * mask_aureg
                loss_2_mean = 0
                if mask_aureg.sum() != 0:
                    loss_2_mean = loss_2_mask.sum() / mask_aureg.sum()

                ## ----- LOSS 3: RECONSTRUCTION -----
                loss_3 = self.criterion(self.decoder(inputs_lstm), labels_reconstruction_pad).sum(dim=-1)
                mask_reconstruction = torch.arange(max(lengths_reconstruction)).expand(len(lengths_reconstruction), max(lengths_reconstruction)) < lengths_reconstruction.unsqueeze(1) # shape: (batch_size, max_seq_len_out)
                mask_reconstruction = mask_reconstruction.to(loss_3.device)
                loss_3_mask = loss_3 * mask_reconstruction
                loss_3_mean = 0
                if mask_reconstruction.sum() != 0:
                    loss_3_mean = loss_3_mask.sum() / mask_reconstruction.sum()

                loss_mean = loss_1_mean + loss_2_mean + loss_3_mean
                loss_total_val_log += loss_mean.item()


                ## ----- SCORE -----
                # # unpad the output sequences
                # output_teafo_unpad = [out[:len_real] for out, len_real in zip(output_teafo_pad_de, lengths_teafo)]
                # output_aureg_unpad = [out[:len_real] for out, len_real in zip(output_aureg_pad_de, lengths_aureg)]
                # merge two sequences
                # predicted_seq = self.concat_output_seq(output_teafo_unpad, output_aureg_unpad)

                # # unpad the label sequences
                # label_teafo_unpad = [seq[:len_real] for seq, len_real in zip(labels_teafo_pad, lengths_teafo)]
                # label_aureg_unpad = [seq[:len_real] for seq, len_real in zip(labels_aureg_pad, lengths_aureg)]
                # label_seq = self.concat_output_seq(label_teafo_unpad, label_aureg_unpad)

                (mean_ade_entire_b, var_ade_entire_b), \
                (mean_ade_future_b, var_ade_future_b), \
                (mean_nade_entire_b, var_nade_entire_b), \
                (mean_nade_future_b, var_nade_future_b), \
                (mean_final_step_err_b, var_fe_b), \
                (mean_capture_success_rate_b, var_capture_success_rate) \
                = self.utils.score_all_predictions(output_teafo_pad_de, labels_teafo_pad, lengths_teafo, 
                                                    output_aureg_pad_de, labels_aureg_pad, lengths_aureg,
                                                    capture_thres=0.1)
                # because MSE divides by batch_size, but we need to sum all to get the total loss and calculate the mean value ourselves, 
                # so in each batch we need to multiply by batch_size
                current_batch_size = len(inputs)

                # sum_mse_all += batch_mean_mse_all*current_batch_size
                # sum_mse_xyz += batch_mean_mse_xyz*current_batch_size
                sum_ade_entire += mean_ade_entire_b*current_batch_size
                sum_ade_future += mean_ade_future_b*current_batch_size
                sum_nade_entire += mean_nade_entire_b*current_batch_size
                sum_nade_future += mean_nade_future_b*current_batch_size
                sum_final_step_err += mean_final_step_err_b*current_batch_size
                sum_capture_success_rate += mean_capture_success_rate_b*current_batch_size
        
        # get mean value of scored data
        mean_loss_total_val_log = loss_total_val_log/len(dl_val.dataset)
        mean_ade_entire = sum_ade_entire/len(dl_val.dataset)
        mean_ade_future = sum_ade_future/len(dl_val.dataset)
        mean_nade_entire = sum_nade_entire/len(dl_val.dataset)
        mean_nade_future = sum_nade_future/len(dl_val.dataset)
        mean_final_step_err = sum_final_step_err/len(dl_val.dataset)
        capture_success_rate = sum_capture_success_rate/len(dl_val.dataset)
        
        return mean_loss_total_val_log, mean_ade_entire, mean_ade_future, mean_nade_entire, mean_nade_future, mean_final_step_err, capture_success_rate

    def concat_output_seq(self, out_seq_teafo, out_seq_aureg):
        # Nối 2 chuỗi đầu ra (dọc theo chiều thời gian - dim=1)
        concatenated_seq = []
        for seq_teafo, seq_aureg in zip(out_seq_teafo, out_seq_aureg):
            # Nối mỗi cặp chuỗi của teacher forcing và autoregressive
            concatenated = torch.cat([seq_teafo, seq_aureg], dim=1)  # Dim=1 là chiều thời gian
            concatenated_seq.append(concatenated)
        return concatenated_seq

    def load_model(self, model_weights_dir):
        encoder_model_path = self.utils.look_for_file(model_weights_dir, 'encoder_model.pth')
        lstm_model_path = self.utils.look_for_file(model_weights_dir, 'lstm_model.pth')
        decoder_model_path = self.utils.look_for_file(model_weights_dir, 'decoder_model.pth')

        self.encoder.load_state_dict(torch.load(encoder_model_path))
        self.encoder.to(self.device)

        self.vls_lstm.load_state_dict(torch.load(lstm_model_path))
        self.vls_lstm.to(self.device)

        self.decoder.load_state_dict(torch.load(decoder_model_path))
        self.decoder.to(self.device)
    def data_correction_check(self, data_train, data_val, data_test):
        data_collection_checker = RoCatRLDataRawCorrectionChecker()
        print('Checking data correction ...')
        for d_train, d_val, d_test in zip(data_train, data_val, data_test):
            d_train_check = data_collection_checker.check_feature_correction(d_train, data_whose_y_up=True)
            d_val_check = data_collection_checker.check_feature_correction(d_val, data_whose_y_up=True)
            d_test_check = data_collection_checker.check_feature_correction(d_test, data_whose_y_up=True)
            if not d_train_check or not d_val_check or not d_test_check:
                self.util_printer.print_red('Data is incorrect, please check', background=True)
                return False
        self.util_printer.print_green('     Data is correct', background=True)
        return True

def main():
    device = torch.device('cuda')
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # feature_size = 9
    # # 1. Dataset and DataLoader
    # dataset = TimeSeriesDataset(num_samples=100, max_len=15, feature_size=feature_size)

    # data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_prediction_ws/src/nae/data/rllab_dataset_no_orientation/data_enrichment/big_plane/big_plane_enrich_for_training'
    # thrown_object = 'big_plane'

    # data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_prediction_ws/src/nae/data/nae_paper_dataset/new_data_format/bamboo/split/bamboo'
    # thrown_object = 'bamboo'
    
    data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_prediction_ws/src/nae/data/nae_paper_dataset/new_data_format/bottle/split/bottle'
    thrown_object = 'bottle'

    checkout_path = None
    wdb_run_id=None   # 't5nlloi0'
    wdb_resume=None   # 'allow'

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
        'lr': 0.0002
    }

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
    nae.util_printer.print_green('Start training ...', background=True)
    wdb_notes = f'NAE_DYNAMIC - {model_params["num_layers_lstm"]} LSTM layers, {model_params["hidden_size"]} hidden size, lr={model_params["lr"]}, batch_size={training_params["batch_size_train"]}'
    nae.init_wandb('nae', run_id=wdb_run_id, resume=wdb_resume, wdb_notes=wdb_notes)
    saved_model_dir = nae.train(data_train, data_val, checkpoint_path=checkout_path)


if __name__ == '__main__':
    main()