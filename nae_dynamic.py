import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as TorchDataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import random
import wandb
from datetime import datetime
import os
import time
import re
import json

from python_utils.printer import Printer
from python_utils.plotter import Plotter

from nae_core.utils.utils import NAE_Utils
from nae_core.utils.submodules.training_utils.data_loader import DataLoader as NAEDataLoader
from nae_core.utils.submodules.training_utils.input_label_generator import InputLabelGenerator
from nae_core.utils.submodules.preprocess_utils.data_preprocess import DataPreprocess
from nae_core.utils.submodules.preprocess_utils.data_preprocess import DataDenormalizer


import logging
import psutil  # Thư viện để theo dõi CPU, RAM
from datetime import datetime
import sys
import traceback
from torch.cuda.amp import autocast, GradScaler

printer_global = Printer()

def log_resources(epoch):
    """Ghi log thông tin tài nguyên hệ thống."""
    # GPU Memory
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3
        logging.info(f"GPU Memory")
        percentage = gpu_memory_allocated / gpu_memory_reserved * 100
        logging.info(f"     Allocated: {gpu_memory_allocated:.2f} GB - {percentage:.2f}%")
        logging.info(f"     Reserved: {gpu_memory_reserved:.2f} GB")

    else:
        logging.info("GPU not available.")

    # CPU and RAM
    cpu_percent = psutil.cpu_percent(interval=1)
    ram_usage = psutil.virtual_memory()
    logging.info(f"     CPU Usage: {cpu_percent}%")
    logging.info(f"     RAM Usage: {ram_usage.percent}%, RAM Available: {ram_usage.available / 1024**3:.2f} GB")




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
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super(VLSLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_rate)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x, lengths_in, lengths_aureg, mask_aureg):
        this_batch_size = x.size(0)
        packed_x = pack_padded_sequence(x, lengths_in, batch_first=True, enforce_sorted=False)
        hi = torch.zeros(self.num_layers, this_batch_size, self.hidden_size).to(x.device)
        ci = torch.zeros(self.num_layers, this_batch_size, self.hidden_size).to(x.device)
        outputs_teafo_unpad = [] # teacher forcing output sequences

        #----- teacher forcing -----
        outputs, (hi, ci) = self.lstm(packed_x, (hi, ci))
        # unpack and only keep real steps, not the padding based on lengths_teacher_forcing
        # Unpack dữ liệu packed outputs
        outputs_teafo_pad, len_real_teafo = pad_packed_sequence(outputs, batch_first=True)
        # Slicing để lấy từng chuỗi theo độ dài thực
        batch_indices = torch.arange(outputs_teafo_pad.size(0))  # [0, 1, 2]
        outputs_teafo_unpad = [outputs_teafo_pad[i, :len_real_teafo[i], :] for i in batch_indices]

        #----- autoregressive -----
        # 1. Lấy output cuối cùng của teacher forcing
        input_aureg_init = [out[-1] for out in outputs_teafo_unpad]
        input_aureg_init = torch.stack(input_aureg_init, dim=0).unsqueeze(1)  # (batch_size, 1, hidden_size)
        outputs_aureg_unpad = self.auto_regressive_loop(input_aureg_init, hi, ci, lengths_aureg, mask_aureg)
        # padding outputs_aureg_unpad
        outputs_aureg_pad = pad_sequence(outputs_aureg_unpad, batch_first=True) 
        return outputs_teafo_pad, outputs_aureg_pad
    
    # # có thể IMPROVE: sử dụng hàm này thay cho hàm trên và kiểm tra sự khác nhau về kết quả và tốc độ
    # def forward(self, x, lengths_in, mask_in, lengths_aureg, mask_aureg):
    #     this_batch_size = x.size(0)
    #     packed_x = pack_padded_sequence(x, lengths_in, batch_first=True, enforce_sorted=False)
    #     hi = torch.zeros(self.num_layers, this_batch_size, self.hidden_size).to(x.device)
    #     ci = torch.zeros(self.num_layers, this_batch_size, self.hidden_size).to(x.device)

    #     #----- teacher forcing -----
    #     outputs, (hi, ci) = self.lstm(packed_x, (hi, ci))
    #     outputs_teafo_pad, _ = pad_packed_sequence(outputs, batch_first=True)

    #     #----- autoregressive -----
    #     # 1. Lấy output cuối cùng của teacher forcing
    #     last_indices = (lengths_in - 1).to(outputs_teafo_pad.device)  # (batch_size,)
    #     input_aureg_init = outputs_teafo_pad[torch.arange(outputs_teafo_pad.size(0), device=outputs_teafo_pad.device), last_indices]

    #     # Chuyển thành (batch_size, 1, hidden_size)
    #     input_aureg_init = input_aureg_init.unsqueeze(1)

    #     # 2. Truyền vào autoregressive loop
    #     outputs_aureg_unpad = self.auto_regressive_loop(input_aureg_init, hi, ci, lengths_aureg, mask_aureg)

    #     # 3. Padding outputs_aureg_unpad để trả về dạng pad
    #     outputs_aureg_pad = pad_sequence(outputs_aureg_unpad, batch_first=True)

    #     return outputs_teafo_pad, outputs_aureg_pad
    
    def auto_regressive_loop(self, input_aureg_init, hi, ci, lengths_aureg, mask_aureg):
        batch_size = input_aureg_init.size(0)
        max_len = lengths_aureg.max().item()

        if mask_aureg.shape[1] != max_len:
            printer_global.print_red(f'VLSLSTM - mask_aureg.shape[1] != max_len -> exit')
            print('     mask_aureg shape: ', mask_aureg.shape)
            print('     max_len: ', max_len)
            raise ValueError("mask_aureg.shape[1] != max_len")
            return

        # output_seq = torch.zeros(batch_size, max_len, hidden_size).to(input_aureg_init[0].device) # -> hardcore calculate
        # create output_seq with dtype of input_aureg_init
        output_seq = torch.zeros(batch_size, max_len, self.lstm.hidden_size, device=input_aureg_init.device, dtype=input_aureg_init.dtype)        # IMPROVE 1: Use dtype of input_aureg_init                  -> save 16.6% training time 12.5s/epoch -> 10.4s/epoch

        # Lấy bước đầu tiên làm đầu vào ban đầu
        lstm_input = input_aureg_init  # Kích thước (batch_size, 1, feature_size)
        # Duyệt qua từng bước thời gian
        for t in range(max_len):
            # current_mask = mask_aureg[:, t]  # Mask tại bước thời gian t   
            current_mask = torch.nonzero(mask_aureg[:, t], as_tuple=True)[0]  # Chỉ số hợp lệ cho batch tại bước t                      # IMPROVE 2: Use torch.nonzero -> save 10% training time    -> save 20% training time 10.4s/epoch -> 8.4s/epoch
            if current_mask.shape[0] == 0:  # Nếu tất cả đều là padding, dừng lại
                logging.error("          VLSLSTM - current_mask.shape[0] == 0 -> exit")
                printer_global.print_red(f'There might be some error in mask making process')
                print(f'     current_mask.shape[0] == 0 -> exit. t: {t}, max_len: {max_len}')
                print('     mask_aureg shape: ', mask_aureg.shape)
                print('     max_len: ', max_len)
                raise ValueError("There might be some error in mask making process")
                break

            # Bỏ những phần tử được padding dựa vào mask, các tensor sẽ được rút ngắn lại theo chiều batch_size
            lstm_input = lstm_input[current_mask]  # (num_real_sequences, 1, feature_size)
            hi_current = hi[:, current_mask, :]  # (num_layers, num_real_sequences, hidden_size)
            ci_current = ci[:, current_mask, :]  # (num_layers, num_real_sequences, hidden_size)

            # Truyền qua LSTM
            output, (hi_new, ci_new) = self.lstm(lstm_input, (hi_current, ci_current))
            # Cập nhật trạng thái ẩn cho các chuỗi thực
            hi[:, current_mask, :] = hi_new
            ci[:, current_mask, :] = ci_new

            # Lưu output vào tensor output_seq
            temp_output = torch.zeros(batch_size, 1, self.lstm.hidden_size, device=output.device, dtype=output.dtype)
            temp_output[current_mask] = output  # Ghi output của các chuỗi thực # chỉ những chuỗi có mask = 1 mới được gán giá trị, còn lại vẫn giữ nguyên giá trị 0
            # Cập nhật đầu vào cho bước tiếp theo
            lstm_input = temp_output[:, -1:, :]  # Sử dụng output hiện tại làm input tiếp theo
            # output_seq.append(temp_output[current_mask])  # Thêm vào danh sách output
            output_seq[:, t:t+1, :] = temp_output
        
        # filter padding and save to a list
        output_seq_final = [output_seq[i, :lengths_aureg[i], :] for i in range(batch_size)]

        return output_seq_final

    # def auto_regressive_loop(self, input_aureg_init, hi, ci, lengths_aureg, mask_aureg):
    #     """
    #     Thực hiện dự đoán autoregressive.
        
    #     Args:
    #         input_aureg_init: Tensor đầu vào ban đầu, kích thước (batch_size, 1, hidden_size).
    #         hi: Trạng thái ẩn ban đầu của LSTM, kích thước (num_layers, batch_size, hidden_size).
    #         ci: Trạng thái ô nhớ ban đầu của LSTM, kích thước (num_layers, batch_size, hidden_size).
    #         lengths_aureg: Độ dài thực của từng chuỗi trong batch, kích thước (batch_size,).
    #         mask_aureg: Mask để xác định các bước thời gian hợp lệ, kích thước (batch_size, max_seq_len).
        
    #     Returns:
    #         output_seq: Tensor đầu ra của mô hình, kích thước (batch_size, max_seq_len, hidden_size).
    #     """
    #     batch_size = input_aureg_init.size(0)
    #     max_len = lengths_aureg.max().item()
    #     hidden_size = self.lstm.hidden_size

    #     # Khởi tạo tensor lưu trữ toàn bộ output (bao gồm cả padding)
    #     output_seq = torch.zeros(batch_size, max_len, hidden_size).to(input_aureg_init.device)

    #     # Đầu vào ban đầu
    #     lstm_input = input_aureg_init  # Kích thước (batch_size, 1, hidden_size)

    #     # Duyệt qua từng bước thời gian
    #     for t in range(max_len):
    #         # Truyền qua LSTM (toàn bộ batch, bao gồm cả padding)
    #         output, (hi, ci) = self.lstm(lstm_input, (hi, ci))

    #         # Lưu output vào tensor (bao gồm cả padding)
    #         output_seq[:, t:t+1, :] = output

    #         # Cập nhật đầu vào cho bước tiếp theo
    #         # Chỉ giữ lại các giá trị từ bước thời gian t hợp lệ (theo mask)
    #         lstm_input = output * mask_aureg[:, t].unsqueeze(1).unsqueeze(2)  # Mask tại bước t
        
    #     # Trích xuất các chuỗi hợp lệ dựa trên lengths_aureg (bỏ padding)
    #     output_seq_final = [output_seq[i, :lengths_aureg[i], :] for i in range(batch_size)]

    #     return output_seq_final

#------------------------- TRAINING -------------------------
class NAEDynamicLSTM():
    def __init__(self, input_size, hidden_size, output_size, num_layers_lstm, lr, 
                 num_epochs, batch_size_train, save_interval, thrown_object, train_id, dropout_rate, warmup_steps=0, loss1_weight=1.0, loss2_weight=1.0, loss2_1_weight=0.0, weight_decay=1e-4,
                 data_step_start=1, data_step_end=-1, data_increment=1,
                 device=torch.device('cuda'),
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
        self.num_epochs = num_epochs
        self.batch_size_train = batch_size_train
        self.save_interval = save_interval
        self.thrown_object = thrown_object + '_model'
        self.train_id = train_id
        self.dropout_rate = dropout_rate
        self.warmup_steps = warmup_steps
        self.loss1_weight = loss1_weight
        self.loss2_weight = loss2_weight
        self.loss2_1_weight = loss2_1_weight
        self.weight_decay = weight_decay


        # create config dict
        self.model_config = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'output_size': output_size,
            'num_layers_lstm': num_layers_lstm,
            'lr': lr,
            'num_epochs': num_epochs,
            'batch_size_train': batch_size_train,
            'save_interval': save_interval,
            'thrown_object': thrown_object,
            'train_id': train_id,
            'dropout_rate': dropout_rate,
            'warmup_steps': warmup_steps,
            'loss1_weight': loss1_weight,
            'loss2_weight': loss2_weight,
            'loss2_1_weight': loss2_1_weight,
            'weight_decay': weight_decay,
            'data_dir': data_dir,
            'data_step_start': data_step_start,
            'data_step_end': data_step_end,
            'data_increment': data_increment,
        }


        self.data_dir = data_dir

        self.run_name = f"{train_id}-{thrown_object}-model_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}_hiddensize{hidden_size}"

        # get dir of folder including this script
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        self.model_dir = os.path.join(cur_dir, 'models', self.thrown_object, self.run_name)

        # Initialize model, loss, optimizer
        self.encoder = Encoder(input_size, hidden_size).to(device)
        self.vls_lstm = VLSLSTM(hidden_size, hidden_size, num_layers_lstm, dropout_rate=dropout_rate).to(device)
        self.decoder = Decoder(hidden_size, output_size).to(device)

        self.initialize_weights(self.encoder)
        self.initialize_weights(self.vls_lstm)
        self.initialize_weights(self.decoder)

        # self.criterion = nn.MSELoss(reduction='none').to(device)  # NOTE: Reduction 'none' to apply masking, default is 'mean'
        # self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.vls_lstm.parameters()) + list(self.decoder.parameters()), lr=lr)
        
        self.weight_decay = weight_decay
        # Định nghĩa các tham số không áp dụng Weight Decay (bias, LayerNorm.weight)
        no_decay = ['bias', 'LayerNorm.weight']
        # Tách các tham số
        params = [
            # Các tham số có áp dụng Weight Decay
            {'params': [p for n, p in self.encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in self.vls_lstm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in self.decoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},

            # Các tham số không áp dụng Weight Decay
            {'params': [p for n, p in self.encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.vls_lstm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.decoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        # Sử dụng AdamW với các nhóm tham số đã tách
        self.optimizer = optim.AdamW(params, lr=lr)
        
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.warmup_lr_scheduler)

        self.util_printer = Printer()
        self.util_plotter = Plotter()
        self.wandb_run_url = ''
        # self.data_preprocess = DataPreprocess()

        self.enable_denormalize = self.check_normalization_files_exist(data_dir)
        if self.enable_denormalize is not None:
            pos_scaler_path = self.enable_denormalize['position_scaler.save']
            vel_scaler_path = self.enable_denormalize['velocity_scaler.save']
            acc_scaler_path = self.enable_denormalize['acceleration_scaler.save']
            self.data_denormalizer = DataDenormalizer(pos_scaler_path=pos_scaler_path, vel_scaler_path=vel_scaler_path, acc_scaler_path=acc_scaler_path)
            self.enable_denormalize = True
            printer_global.print_green('Enable denormalize', background=True); input('Press Enter to continue')
        else:
            self.enable_denormalize = False

        print('\n')
        printer_global.print_green('----------------- MODEL PARAMS NUMBER -----------------', background=True)
        print('     Parameters number of encoder: ', self.utils.count_parameters(self.encoder))
        print('     Parameters number of LSTM: ', self.utils.count_parameters(self.vls_lstm))
        print('     Parameters number of decoder: ', self.utils.count_parameters(self.decoder))
        print('     Total number of parameters: ', self.utils.count_parameters(self.encoder) + self.utils.count_parameters(self.vls_lstm) + self.utils.count_parameters(self.decoder))


    def warmup_lr_scheduler(self, epoch):
        if epoch < self.warmup_steps:
            ratio = (epoch + 1) / self.warmup_steps
            print('     - Warm-up ratio: ', ratio)
            return ratio  # Tăng dần từ 0 đến 1
        return 1.0  # Sau warm-up, giữ nguyên learning rate

    def collate_pad_fn(self, batch):
        inputs, labels_teafo, labels_aureg, label_reconstruction = zip(*batch)

        lengths_in = [len(seq) for seq in inputs]
        lengths_teafo = [len(seq) for seq in labels_teafo]
        lengths_aureg = [len(seq) for seq in labels_aureg]
        lengths_reconstruction = [len(seq) for seq in label_reconstruction]

        # move to device
        lengths_in = torch.tensor(lengths_in, dtype=torch.int64)
        lengths_teafo= torch.tensor(lengths_teafo, dtype=torch.int64)
        lengths_aureg= torch.tensor(lengths_aureg, dtype=torch.int64)
        lengths_reconstruction= torch.tensor(lengths_reconstruction, dtype=torch.int64)

        # Padding sequences to have same length in batch
        inputs_pad = pad_sequence(inputs, batch_first=True)  # Shape: (batch_size, max_seq_len_in, feature_dim)
        labels_teafo_pad = pad_sequence(labels_teafo, batch_first=True)
        labels_aureg_pad = pad_sequence(labels_aureg, batch_first=True)
        labels_reconstruction_pad = pad_sequence(label_reconstruction, batch_first=True)

        # Tạo mask dựa trên độ dài của chuỗi
        mask_in = torch.arange(lengths_in.max().item()).expand(len(lengths_in), lengths_in.max().item()) < lengths_in.unsqueeze(1) # shape: (batch_size, max_seq_len_in)
        mask_teafo = torch.arange(lengths_teafo.max().item()).expand(len(lengths_teafo), lengths_teafo.max().item()) < lengths_teafo.unsqueeze(1) # shape: (batch_size, max_seq_len_out)
        mask_aureg = torch.arange(lengths_aureg.max().item()).expand(len(lengths_aureg), lengths_aureg.max().item()) < lengths_aureg.unsqueeze(1) # shape: (batch_size, max_seq_len_out)
        mask_reconstruction = torch.arange(lengths_reconstruction.max().item()).expand(len(lengths_reconstruction), lengths_reconstruction.max().item()) < lengths_reconstruction.unsqueeze(1) # shape: (batch_size, max_seq_len_out)
        
        
        # # check if the mask is correct:
        # if self.utils.has_all_zero_rows(mask_in)[0]:
        #     print('     mask_in has all zero rows')
        #     input()
        # if self.utils.has_all_zero_rows(mask_teafo)[0]:
        #     print('     mask_teafo has all zero rows')
        #     input()
        # if self.utils.has_all_zero_rows(mask_aureg)[0]:
        #     print('     mask_aureg has all zero rows')
        #     input()
        # if self.utils.has_all_zero_rows(mask_reconstruction)[0]:
        #     print('     mask_reconstruction has all zero rows')
        #     input()

        return (
            (inputs_pad, lengths_in, mask_in),
            (labels_teafo_pad, lengths_teafo, mask_teafo),
            (labels_aureg_pad, lengths_aureg, mask_aureg),
            (labels_reconstruction_pad, lengths_reconstruction, mask_reconstruction),
        )

    def initialize_weights(self, model):
        for layer in model.modules():
            # Khởi tạo lớp Linear (FC)
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)  # Xavier Uniform cho trọng số
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)  # Bias thường khởi tạo là 0
                    
            # Khởi tạo lớp LSTM
            elif isinstance(layer, nn.LSTM):
                for name, param in layer.named_parameters():
                    if 'weight_ih' in name:  # Input-hidden weights
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:  # Hidden-hidden weights
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)  # Bias thường khởi tạo là 0
                        # Hoặc khởi tạo đặc biệt cho phần bias của gate forget
                        gate_size = param.size(0) // 4
                        param.data[gate_size:2 * gate_size] = 1.0  # Forget gate bias = 1.0
    
    def train(self, data_train, data_val, checkpoint_path=None, enable_wandb=False,  save_model=True, test_anomaly=False, test_cuda_blocking=False, logging_level=logging.WARNING, debug=False, grad_norm_clip=None):
        # self._init_logging(logging_level, test_anomaly, test_cuda_blocking)
        scaler = GradScaler()
        if checkpoint_path:
            start_epoch = self.load_checkpoint(checkpoint_path) + 1
        else:
            start_epoch = 0
        try:

            # For debugging
            if test_anomaly:
                # logging.info("Testing torch.autograd.set_detect_anomaly(True)") if debug else None
                torch.autograd.set_detect_anomaly(True)
            if test_cuda_blocking:
                # logging.info("Testing CUDA_LAUNCH_BLOCKING") if debug else None
                os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

            # Khởi tạo thời gian bắt đầu
            start_t = time.time()
            dataloader_train = TorchDataLoader(data_train, batch_size=self.batch_size_train, collate_fn=lambda x: self.collate_pad_fn(x), shuffle=True,
                                                num_workers=8,
                                                pin_memory=True,
                                                persistent_workers=True)

            count_nan_grad = 0
            count_tiny_grad = 0
            count_huge_grad = 0
            count_inf_grad = 0
            for epoch in range(start_epoch, self.num_epochs):
                time_start_epoch = time.time()
                self.encoder.train()
                self.vls_lstm.train()
                self.decoder.train()

                # loss_epoch_log = 0
                loss_total_train_log = 0.0
                loss_1_train_log = 0.0
                loss_2_train_log = 0.0
                loss_3_train_log = 0.0
                loss_all_log = []

                # logging.info(f"\n\n====================     EPOCH {epoch}     ====================")        
                batch_idx = 0
                # time_batch_all = []
                for batch in dataloader_train:
                    # time_batch_start = time.time()
                    # self.util_printer.print_green(f'Batch: {batch_idx}')
                    batch_idx += 1
                    # logging.info(f"\nBatch {batch_idx}:") if debug else None
                    try:
                        # logging.info(f"TRAINING:") if debug else None
                        (inputs_pad, lengths_in, mask_in), \
                        (labels_teafo_pad, lengths_teafo, mask_teafo), \
                        (labels_aureg_pad, lengths_aureg, mask_aureg),\
                        (labels_reconstruction_pad, lengths_reconstruction, mask_reconstruction) = batch

                        inputs_pad = inputs_pad.to(self.device, non_blocking=True).float()
                        labels_teafo_pad = labels_teafo_pad.to(self.device, non_blocking=True)
                        labels_aureg_pad = labels_aureg_pad.to(self.device, non_blocking=True)
                        labels_reconstruction_pad= labels_reconstruction_pad.to(self.device, non_blocking=True)

                        mask_in = mask_in.to(self.device, non_blocking=True)
                        mask_teafo = mask_teafo.to(self.device, non_blocking=True)
                        mask_aureg = mask_aureg.to(self.device, non_blocking=True)
                        mask_reconstruction = mask_reconstruction.to(self.device, non_blocking=True)


                        # # check shape masks
                        # print('OUT:')
                        # print('     mask_in shape: ', mask_in.shape)
                        # print('     mask_teafo shape: ', mask_teafo.shape)
                        # print('     mask_aureg shape: ', mask_aureg.shape)
                        # print('     mask_reconstruction shape: ', mask_reconstruction.shape)
                        # input()
                        # mask_combined = torch.cat([mask_teafo, mask_aureg], dim=1)

                        # time_flag_1 = time.time() # load data time
                        with autocast():
                            inputs_lstm = self.encoder(inputs_pad)
                            outputs_teafo_pad, output_aureg_pad = self.vls_lstm(inputs_lstm, lengths_in, lengths_aureg, mask_aureg)
                            output_teafo_pad_de = self.decoder(outputs_teafo_pad)
                            output_aureg_pad_de = self.decoder(output_aureg_pad)
                            
                            # time_flag_2 = time.time() # forward pass time

                            ##  ----- LOSS 1: TEACHER FORCING -----
                            # loss_1 = self.criterion(output_teafo_pad_de, labels_teafo_pad).sum(dim=-1)  # Shape: (batch_size, max_seq_len_out)
                            # # Tạo mask dựa trên chiều dài thực
                            # loss_1_mask = loss_1 * mask_teafo  # Masked loss
                            # loss_1_mean = 0
                            # if mask_teafo.sum() != 0:
                            #     loss_1_mean = loss_1_mask.sum() / mask_teafo.sum()
                            # else:
                            #     self.util_printer.print_red(f'Loss 1 - mask_teafo.sum() == 0')
                            #     input('DEBUG')

                            loss_1_mean = self.compute_masked_mse_loss(predictions=output_teafo_pad_de, labels=labels_teafo_pad, mask=mask_teafo)

                            ## ----- LOSS 2: AUTOREGRESSIVE -----
                            # loss_2 = self.criterion(output_aureg_pad_de, labels_aureg_pad).sum(dim=-1)
                            # loss_2_mask = loss_2 * mask_aureg
                            # loss_2_mean = 0
                            # if mask_aureg.sum() != 0:
                            #     loss_2_mean = loss_2_mask.sum() / mask_aureg.sum()
                            # else:
                            #     self.util_printer.print_red(f'Loss 2 - mask_aureg.sum() == 0')
                            #     input('DEBUG')

                            loss_2_mean = self.compute_masked_mse_loss(predictions=output_aureg_pad_de, labels=labels_aureg_pad, mask=mask_aureg)


                            ## ----- LOSS 2-1: Last point error -----
                            # mask_aureg_int = mask_aureg.int()
                            # last_indices = mask_aureg_int.flip(dims=[1]).argmax(dim=1)
                            # last_indices = mask_aureg_int.size(1) - 1 - last_indices
                            # mask_last_point = torch.zeros_like(mask_aureg_int, dtype=torch.bool)
                            # mask_last_point[torch.arange(mask_aureg_int.size(0), device=mask_aureg_int.device), last_indices] = 1
                            # loss_2_1_mask = loss_2 * mask_last_point
                            # loss_2_1_mean = 0
                            # if mask_last_point.sum() != 0:
                            #     loss_2_1_mean = loss_2_1_mask.sum() / mask_last_point.sum()
                            # else:
                            #     self.util_printer.print_red(f'Loss 2-1 - mask_last_point.sum() == 0')
                            #     input('DEBUG')


                            ## ----- LOSS 3: RECONSTRUCTION -----
                            # loss_3 = self.criterion(self.decoder(inputs_lstm), labels_reconstruction_pad).sum(dim=-1)
                            # loss_3_mask = loss_3 * mask_reconstruction
                            # loss_3_mean = 0
                            # if mask_reconstruction.sum() != 0:
                            #     loss_3_mean = loss_3_mask.sum() / mask_reconstruction.sum()
                            # else:
                            #     self.util_printer.print_red(f'Loss 3 - mask_reconstruction.sum() == 0')
                            #     input('DEBUG')

                            loss_3_mean = self.compute_masked_mse_loss(predictions=self.decoder(inputs_lstm), labels=labels_reconstruction_pad, mask=mask_reconstruction)

                            
                            # loss_mean = self.loss1_weight*loss_1_mean + self.loss2_weight*loss_2_mean + loss_3_mean + self.loss2_1_weight * loss_2_1_mean
                            loss_mean = self.loss1_weight*loss_1_mean + self.loss2_weight*loss_2_mean + loss_3_mean

                        # Backward pass
                        self.optimizer.zero_grad()
                                                
                        # loss_mean.backward()

                        scaler.scale(loss_mean).backward()



                        models = [self.encoder, self.vls_lstm, self.decoder]
                        total_gradient_norm = self.compute_total_gradient_norm(models)
                        if total_gradient_norm == float('nan'):
                            self.util_printer.print_purple(f"Epoch {epoch} - Batch {batch_idx}/{len(dataloader_train)} - Total Gradient Norm: {total_gradient_norm}")
                            count_nan_grad += 1
                        if total_gradient_norm < 1e-5:
                            self.util_printer.print_pink(f"Epoch {epoch} - Batch {batch_idx}/{len(dataloader_train)} - Total Gradient Norm: {total_gradient_norm}")
                            count_tiny_grad += 1
                        # check infinities
                        if total_gradient_norm == float('inf'):
                            self.util_printer.print_red(f"Epoch {epoch} - Batch {batch_idx}/{len(dataloader_train)} - Total Gradient Norm: {total_gradient_norm}")
                            count_inf_grad += 1
                        elif total_gradient_norm > 100:
                            self.util_printer.print_yellow(f"Epoch {epoch} - Batch {batch_idx}/{len(dataloader_train)} - Total Gradient Norm: {total_gradient_norm}")
                            count_huge_grad += 1

                        # Clip gradient
                        if grad_norm_clip and grad_norm_clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                list(self.encoder.parameters()) + list(self.vls_lstm.parameters()) + list(self.decoder.parameters()),
                                max_norm=grad_norm_clip
                            )


                        # time_flag_4 = time.time() # backward pass time
                        # self.optimizer.step()

                        scaler.step(self.optimizer)
                        scaler.update()

                        # time_flag_5 = time.time() # optimizer step time

                        # Log loss cho batch
                        loss_total_train_log += loss_mean.item()
                        loss_1_train_log += loss_1_mean.item()
                        loss_2_train_log += loss_2_mean.item()
                        loss_3_train_log += loss_3_mean.item()

                        # torch.cuda.synchronize()
                        # time_batch_total = time.time() - time_batch_start
                        # print(f'    Load data time:     % {(time_flag_1 - time_batch_start)/time_batch_total:.3f}')
                        # print(f'    Forward pass time:  % {(time_flag_2 - time_flag_1)/time_batch_total:.3f}')
                        # print(f'    Loss cal time:      % {(time_flag_3 - time_flag_2)/time_batch_total:.3f}')
                        # print(f'    Backward pass time: % {(time_flag_4 - time_flag_3)/time_batch_total:.3f}')
                        # print(f'    Optimizer step time:% {(time_flag_5 - time_flag_4)/time_batch_total:.3f}')
                        # print(f'    Total batch time: sec {time_batch_total}')
                        # time_batch_all.append(time_batch_total)

                    except RuntimeError as e:
                        # logging.error(f"Epoch {epoch} -      RuntimeError during training!") if debug else None
                        # logging.error(f"Epoch {epoch} -          Error message: %s", str(e)) if debug else None
                        # logging.error(f"Epoch {epoch} -          Stack trace:") if debug else None
                        # logging.error(traceback.format_exc()) if debug else None  # Log stack trace đầy đủ
                        sys.stderr.flush()
                        log_resources(epoch)
                        raise e
                    
                # # calculate mean time per batch
                # time_batch_all = np.array(time_batch_all)
                # time_batch_mean = np.mean(time_batch_all)
                # self.util_printer.print_green(f'Mean time per batch: {time_batch_mean:.3f} s')


                # get average loss over all batches
                loss_total_train_log /= len(dataloader_train)
                loss_1_train_log /= len(dataloader_train)
                loss_2_train_log /= len(dataloader_train)
                loss_3_train_log /= len(dataloader_train)
                loss_all_log.append([loss_1_train_log, loss_2_train_log, loss_3_train_log, loss_total_train_log])

                if epoch % self.save_interval == 0 and save_model:
                    self.save_model(epoch, len(data_train), start_t, loss_all_log)
                time_train_epoch = time.time() - time_start_epoch
                


                # 2. ----- FOR VALIDATION -----
                try:
                    # logging.info(f"VALIDATION:")
                    mean_loss_total_log_val, mean_loss_1_log_val, mean_loss_2_log_val, mean_loss_3_log_val, \
                    (mean_ade_entire,   std_ade_entire), \
                    (mean_ade_future,   std_ade_future), \
                    (mean_ade_past,     std_ade_past), \
                    (mean_nade_entire,  std_nade_entire), \
                    (mean_nade_future,  std_nade_future), \
                    (mean_nade_past,    std_nade_past), \
                    (mean_fe, std_fe, converge_to_final_point_trending), \
                    capture_success_rates = self.validate_and_score(data=data_val, shuffle=False, denorm_data=self.enable_denormalize)
                except RuntimeError as e:
                    # logging.error("Epoch {epoch} - VAL      RuntimeError during validation!") if debug else None
                    # logging.error("Epoch {epoch} - VAL      Error message: %s", str(e)) if debug else None
                    # logging.error("Epoch {epoch} - VAL      Stack trace:") if debug else None
                    # logging.error(traceback.format_exc()) if debug else None
                    raise e
                
                torch.cuda.empty_cache()
                
                time_valid_epoch = time.time() - time_start_epoch - time_train_epoch
                time_total_epoch = time.time() - time_start_epoch   # training speed of this epoch
        
                # 3. ----- FOR WANDB LOG -----
                if enable_wandb:
                    total_training_time = time.time() - start_t
                    wandb.log({
                        'valid_loss_total': mean_loss_total_log_val,
                        'valid_loss_1': mean_loss_1_log_val,
                        'valid_loss_2': mean_loss_2_log_val,
                        'valid_loss_3': mean_loss_3_log_val,
                    }, step=epoch)

                    wandb.log({
                        'valid_mean_ade_entire': mean_ade_entire,
                        'valid_mean_ade_entire_min': mean_ade_entire - std_ade_entire,
                        'valid_mean_ade_entire_max': mean_ade_entire + std_ade_entire,
                    }, step=epoch)

                    wandb.log({
                        'valid_mean_ade_future': mean_ade_future,
                        'valid_mean_ade_future_min': mean_ade_future - std_ade_future,
                        'valid_mean_ade_future_max': mean_ade_future + std_ade_future,
                    }, step=epoch)

                    wandb.log({
                        'valid_mean_ade_past': mean_ade_past,
                        'valid_mean_ade_past_min': mean_ade_past - std_ade_past,
                        'valid_mean_ade_past_max': mean_ade_past + std_ade_past,
                    }, step=epoch)

                    wandb.log({
                        'valid_mean_nade_entire': mean_nade_entire,
                        'valid_mean_nade_entire_min': mean_nade_entire - std_nade_entire,
                        'valid_mean_nade_entire_max': mean_nade_entire + std_nade_entire,
                    }, step=epoch)

                    wandb.log({
                        'valid_mean_nade_future': mean_nade_future,
                        'valid_mean_nade_future_min': mean_nade_future - std_nade_future,
                        'valid_mean_nade_future_max': mean_nade_future + std_nade_future,
                    }, step=epoch)

                    wandb.log({
                        'valid_mean_nade_past': mean_nade_past,
                        'valid_mean_nade_past_min': mean_nade_past - std_nade_past,
                        'valid_mean_nade_past_max': mean_nade_past + std_nade_past,
                    }, step=epoch)

                    wandb.log({
                        "valid_mean_IE_err": mean_fe,
                        "valide_mean_IE_min": mean_fe - std_fe,
                        "valid_mean_IE_max": mean_fe + std_fe,
                    }, step=epoch)

                    for cap_rate_data in capture_success_rates:
                        cap_thr = cap_rate_data[0]
                        cap_rate = cap_rate_data[1]
                        cap_std = cap_rate_data[2]
                        wandb.log({
                            f"valid_capture_success_rate_{cap_thr}": cap_rate,
                            f"valid_capture_success_rate_{cap_thr}_min": cap_rate - cap_std,
                            f"valid_capture_success_rate_{cap_thr}_max": cap_rate + cap_std,
                        }, step=epoch)

                    wandb.log({
                        'converge_to_final_point_trending': converge_to_final_point_trending,
                        'training speed (s/epoch)': time_total_epoch,
                        'training_time_mins': total_training_time/60,
                        'learning_rate': self.optimizer.param_groups[0]["lr"],
                    }, step=epoch)

                if (epoch) % 1 == 0:
                    run_time_total = time.time() - start_t
                    self.util_printer.print_green(f'Epoch [{epoch}/{self.num_epochs}] {run_time_total/60:.3f} mins', background=False)
                    print(f'    Training speed:     {time_total_epoch:.3f} s/epoch')
                    print(f'    Training time left: {(self.num_epochs - (epoch+1)) * time_total_epoch/3600:.2f} hours')
                    print(f'    training time %:    {time_train_epoch/time_total_epoch:.2f}')
                    print(f'    Validation time %:  {time_valid_epoch/time_total_epoch:.2f}')
                    print(f'    Loss:               {loss_total_train_log:.6f}')
                    print(f'    learning rate:      {self.optimizer.param_groups[0]["lr"]}')

                    printer_global.print_yellow(f'    count_nan_grad:      {count_nan_grad}')
                    printer_global.print_yellow(f'    count_tiny_grad:    {count_tiny_grad}')
                    printer_global.print_yellow(f'    count_huge_grad:    {count_huge_grad}')
                    printer_global.print_yellow(f'    count_inf_grad:     {count_inf_grad}')
                    printer_global.print_green(f'    mean_fe +-std:         {mean_fe:.6f} +- {std_fe:.6f}')
                    print('\n-----------------------------------')
                if epoch < self.warmup_steps:
                    self.scheduler.step()

        finally:
            if debug:
                if sys.stdout:
                    sys.stderr.flush()
                if sys.stderr:
                    sys.stderr.flush()

        final_model_dir = self.save_model(epoch, len(data_train), start_t, loss_all_log)

        if enable_wandb:
            wandb.finish()
        return final_model_dir
    
    # def compute_masked_mse_loss(self, predictions, labels, mask):
    #     """
    #     Compute the masked MSE loss for a batch of sequences, skipping fully padded sequences.
    #     """
    #     # Expand mask to match feature dimensions
    #     mask = mask.unsqueeze(-1)  # Shape: (batch_size, max_seq_len, 1)

    #     # Compute squared differences and apply mask
    #     squared_diff = (predictions - labels) ** 2
    #     masked_squared_diff = squared_diff * mask  # Mask padding values

    #     # Sum squared differences and count valid elements
    #     sum_squared_diff = masked_squared_diff.sum(dim=(1, 2))  # Sum over seq_len and feature_dim
    #     valid_count = mask.sum(dim=(1, 2))                      # Count of valid elements

    #     # Create a mask to skip fully padded sequences
    #     valid_mask = valid_count > 0  # Boolean mask: True for valid sequences, False for fully padded ones

    #     # Filter out invalid sequences
    #     sum_squared_diff = sum_squared_diff[valid_mask]  # Only keep valid sequences
    #     valid_count = valid_count[valid_mask]            # Only keep valid counts

    #     # Compute mean squared error per sequence
    #     mse_per_sequence = sum_squared_diff / valid_count

    #     # Average over the remaining valid sequences
    #     batch_mse_loss = mse_per_sequence.mean()

    #     return batch_mse_loss


    def compute_masked_mse_loss(self, predictions, labels, mask):
        """
        Compute the masked MSE loss for a batch of sequences.
        """
        # Expand mask to match feature dimensions
        mask = mask.unsqueeze(-1)  # Shape: (batch_size, max_seq_len, 1)

        # Compute squared differences and apply mask
        squared_diff = (predictions - labels) ** 2
        masked_squared_diff = squared_diff * mask  # Mask padding values

        # Sum squared differences and count valid elements
        sum_squared_diff = masked_squared_diff.sum(dim=(1, 2))  # Sum over seq_len and feature_dim
        valid_count = mask.sum(dim=(1, 2)).clamp(min=1e-8)       # Avoid division by zero

        # Compute mean squared error per sequence and average over batch
        mse_per_sequence = sum_squared_diff / valid_count       # /t in the paper
        batch_mse_loss = mse_per_sequence.mean()                # Mean over batch

        return batch_mse_loss

    def _init_logging(self, logging_level, test_anomaly, test_cuda_blocking):    
        # Save to self.model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        log_filename = f"train_log.log"
        log_filename = os.path.join(self.model_dir, log_filename)
        self.util_printer.print_green(f"Log file: {log_filename}", background=True)

        logging.basicConfig(filename=log_filename, level=logging_level, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.warning("Starting training...")
        logging.warning("  test_anomaly: " + str(test_anomaly))
        logging.warning("  test_cuda_blocking: " + str(test_cuda_blocking))
        logging.warning("---------------------------")

    def init_wandb(self, project_name, run_id=None, resume=None, wdb_notes=''):        
        wandb.init(
            # set the wandb project where this run will be logged
            project = project_name,
            name=self.run_name,
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
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.vls_lstm.load_state_dict(checkpoint['model_state_dict'])
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
        
        # self.utils.save_model_info(self.data_dir, model_dir, data_num, self.num_epochs, self.batch_size_train, start_t, training_t, loss_all_data, self.wandb_run_url)
        self.save_model_config(model_dir)
        print(f'Models were saved to {model_dir}')
        return model_dir

    def validate_and_score(self, data, shuffle=False, inference=False, denorm_data=False):
        self.util_printer.print_green(f'Validating: {len(data)} samples')
        dataloader = TorchDataLoader(data, batch_size=len(data), collate_fn=lambda x: self.collate_pad_fn(x), shuffle=shuffle)
        self.vls_lstm.eval()
        self.encoder.eval()
        self.decoder.eval()

        loss_total_log = 0.0
        loss_1_log = 0.0
        loss_2_log = 0.0
        loss_3_log = 0.0
        # sum_mse_all = 0.0
        # sum_mse_xyz = 0.0
        # mean_ade_entire = None
        # mean_ade_future = None
        # mean_nade_entire = None
        # mean_nade_future = None
        # mean_final_step_err = None
        # mean_capture_success_rate = None
        # converge_to_final_point_trending = None

        if inference:
            predicted_seqs_all = []
            label_seqs_all = []
        with torch.no_grad():
            for batch in dataloader:    # because we feed all data at once so there is only 1 batch
                (inputs_pad, lengths_in, mask_in), \
                (labels_teafo_pad, lengths_teafo, mask_teafo), \
                (labels_aureg_pad, lengths_aureg, mask_aureg),\
                (labels_reconstruction_pad, lengths_reconstruction, mask_reconstruction) = batch

                inputs_pad = inputs_pad.to(self.device).float()
                labels_teafo_pad = labels_teafo_pad.to(self.device)
                labels_aureg_pad = labels_aureg_pad.to(self.device)
                labels_reconstruction_pad= labels_reconstruction_pad.to(self.device)

                mask_in = mask_in.to(self.device)
                mask_teafo = mask_teafo.to(self.device)
                mask_aureg = mask_aureg.to(self.device)
                mask_reconstruction = mask_reconstruction.to(self.device)


                # Predict
                inputs_lstm = self.encoder(inputs_pad)
                outputs_teafo_pad, output_aureg_pad = self.vls_lstm(inputs_lstm, lengths_in, lengths_aureg, mask_aureg)
                output_teafo_pad_de = self.decoder(outputs_teafo_pad)
                output_aureg_pad_de = self.decoder(output_aureg_pad)

                if denorm_data==True:
                    # denormalize outputs, labels
                    output_teafo_pad_de = self.data_denormalizer.denormalize_data_all_feature_on_cuda(output_teafo_pad_de)
                    labels_teafo_pad = self.data_denormalizer.denormalize_data_all_feature_on_cuda(labels_teafo_pad)
                    output_aureg_pad_de = self.data_denormalizer.denormalize_data_all_feature_on_cuda(output_aureg_pad_de)
                    labels_aureg_pad = self.data_denormalizer.denormalize_data_all_feature_on_cuda(labels_aureg_pad)

                if inference:
                    predicted_seqs = self.merge_pad_trajectories(output_teafo_pad_de, lengths_teafo, output_aureg_pad_de, lengths_aureg)
                    label_seqs = self.merge_pad_trajectories(labels_teafo_pad, lengths_teafo, labels_aureg_pad, lengths_aureg)
                    
                    predicted_seqs_all.extend(predicted_seqs)
                    label_seqs_all.extend(label_seqs)
                    # continue
                
                ##  ----- LOSS 1: TEACHER FORCING -----
                # loss_1 = self.criterion(output_teafo_pad_de, labels_teafo_pad).sum(dim=-1)  # Shape: (batch_size, max_seq_len_out)
                # # Tạo mask dựa trên chiều dài thực
                # loss_1_mask = loss_1 * mask_teafo  # Masked loss
                # loss_1_mean = 0
                # if mask_teafo.sum() != 0:
                #     loss_1_mean = loss_1_mask.sum() / mask_teafo.sum()

                loss_1_mean = self.compute_masked_mse_loss(predictions=output_teafo_pad_de, labels=labels_teafo_pad, mask=mask_teafo)

                ## ----- LOSS 2: AUTOREGRESSIVE -----
                # loss_2 = self.criterion(output_aureg_pad_de, labels_aureg_pad).sum(dim=-1)
                # loss_2_mask = loss_2 * mask_aureg
                # loss_2_mean = 0
                # if mask_aureg.sum() != 0:
                #     loss_2_mean = loss_2_mask.sum() / mask_aureg.sum()

                loss_2_mean = self.compute_masked_mse_loss(predictions=output_aureg_pad_de, labels=labels_aureg_pad, mask=mask_aureg)

                ## ----- LOSS 3: RECONSTRUCTION -----
                # loss_3 = self.criterion(self.decoder(inputs_lstm), labels_reconstruction_pad).sum(dim=-1)
                # loss_3_mask = loss_3 * mask_reconstruction
                # loss_3_mean = 0
                # if mask_reconstruction.sum() != 0:
                #     loss_3_mean = loss_3_mask.sum() / mask_reconstruction.sum()

                loss_3_mean = self.compute_masked_mse_loss(predictions=self.decoder(inputs_lstm), labels=labels_reconstruction_pad, mask=mask_reconstruction)

                if torch.isnan(loss_1_mean) or torch.isinf(loss_1_mean):
                    print("loss_1_mean contains NaN or Infinity!")
                    # logging.error(f"Epoch {epoch} - loss_1_mean contains NaN or Infinity!") if debug else None

                # Kiểm tra loss_2_mean
                if torch.isnan(loss_2_mean) or torch.isinf(loss_2_mean):
                    print("loss_2_mean contains NaN or Infinity!")
                    # logging.error(f"Epoch {epoch} - loss_2_mean contains NaN or Infinity!") if debug else None

                # Kiểm tra loss_3_mean
                if torch.isnan(loss_3_mean) or torch.isinf(loss_3_mean):
                    print("loss_3_mean contains NaN or Infinity!")
                    # logging.error(f"Epoch {epoch} - loss_3_mean contains NaN or Infinity!") if debug else None
                    
                loss_mean = loss_1_mean + loss_2_mean + loss_3_mean

                # Kiểm tra NaN
                if torch.isnan(loss_mean):
                    # logging.error(f"Epoch {epoch} -      Loss contains NaN!") if debug else None
                    raise ValueError("     NaN detected in loss!")
                loss_total_log += loss_mean.item()
                loss_1_log += loss_1_mean.item()
                loss_2_log += loss_2_mean.item()
                loss_3_log += loss_3_mean.item()

                ## ----- SCORE -----
                (mean_ade_entire,   std_ade_entire), \
                (mean_ade_future,   std_ade_future), \
                (mean_ade_past,     std_ade_past), \
                (mean_nade_entire,  std_nade_entire), \
                (mean_nade_future,  std_nade_future), \
                (mean_nade_past,    std_nade_past), \
                (mean_fe, std_fe, converge_to_final_point_trending), \
                capture_success_rates \
                = self.utils.score_all_predictions(output_teafo_pad_de, labels_teafo_pad, lengths_teafo, mask_teafo, 
                                                    output_aureg_pad_de, labels_aureg_pad, lengths_aureg, mask_aureg,
                                                    capture_thres=[0.01, 0.05, 0.1], debug=inference)
                
                # print('-----------------converge_to_final_point_trending: ', converge_to_final_point_trending)
        
        if inference:
            return predicted_seqs_all, label_seqs_all, converge_to_final_point_trending
        # get mean value of scored data
        mean_loss_total_log = loss_total_log/len(dataloader.dataset)
        mean_loss_1_log = loss_1_log/len(dataloader.dataset)
        mean_loss_2_log = loss_2_log/len(dataloader.dataset)
        mean_loss_3_log = loss_3_log/len(dataloader.dataset)
        
        return  mean_loss_total_log, mean_loss_1_log, mean_loss_2_log, mean_loss_3_log, \
                (mean_ade_entire,   std_ade_entire), \
                (mean_ade_future,   std_ade_future), \
                (mean_ade_past,     std_ade_past), \
                (mean_nade_entire,  std_nade_entire), \
                (mean_nade_future,  std_nade_future), \
                (mean_nade_past,    std_nade_past), \
                (mean_fe, std_fe, converge_to_final_point_trending), \
                capture_success_rates

    def concat_output_seq(self, out_seq_teafo, out_seq_aureg):
        # Nối 2 chuỗi đầu ra (dọc theo chiều thời gian - dim=1)
        out_seq_teafo = [out.cpu().numpy() for out in out_seq_teafo]
        out_seq_aureg = [out.cpu().numpy() for out in out_seq_aureg]
        concatenated_seq = []
        for seq_teafo, seq_aureg in zip(out_seq_teafo, out_seq_aureg):
            # Nối mỗi cặp chuỗi của teacher forcing và autoregressive
            # concatenated = torch.cat([seq_teafo, seq_aureg], dim=1)  # Dim=1 là chiều thời gian
            seq_teafo = np.array(seq_teafo)
            seq_aureg = np.array(seq_aureg)
            # print('check seq_teafo shape: ', seq_teafo.shape)
            # print('check seq_aureg shape: ', seq_aureg.shape)
            concatenated = np.concatenate([seq_teafo, seq_aureg], axis=0)
            # print('check concatenated shape: ', concatenated.shape)
            # input()
            concatenated_seq.append(concatenated)
            
        return concatenated_seq

    def extract_epoch_idx(self, path):
        # Sử dụng regex để tìm số epoch từ đường dẫn
        match = re.search(r'epochs(\d+)_', path)
        if match:
            return int(match.group(1))
        return None
    
    def load_model(self, model_weights_dir, weights_only=False):
        encoder_model_path = self.utils.look_for_file(model_weights_dir, 'encoder_model.pth')
        lstm_model_path = self.utils.look_for_file(model_weights_dir, 'lstm_model.pth')
        decoder_model_path = self.utils.look_for_file(model_weights_dir, 'decoder_model.pth')

        self.encoder.load_state_dict(torch.load(encoder_model_path, weights_only=weights_only))
        self.encoder.to(self.device)

        self.vls_lstm.load_state_dict(torch.load(lstm_model_path, weights_only=weights_only))
        self.vls_lstm.to(self.device)

        self.decoder.load_state_dict(torch.load(decoder_model_path, weights_only=weights_only))
        self.decoder.to(self.device)
        epoch_idx = self.extract_epoch_idx(model_weights_dir)
        return epoch_idx
    
    # def data_correction_check(self, data_train, data_val, data_test):
    #     data_collection_checker = RoCatRLDataRawCorrectionChecker()
    #     print('Checking data correction ...')
    #     for d_train, d_val, d_test in zip(data_train, data_val, data_test):
    #         d_train_check = data_collection_checker.check_feature_correction(d_train, data_whose_y_up=True)
    #         d_val_check = data_collection_checker.check_feature_correction(d_val, data_whose_y_up=True)
    #         d_test_check = data_collection_checker.check_feature_correction(d_test, data_whose_y_up=True)
    #         if not d_train_check or not d_val_check or not d_test_check:
    #             self.util_printer.print_red('Data is incorrect, please check', background=True)
    #             return False
    #     self.util_printer.print_green('     Data is correct', background=True)
    #     return True
    
    def merge_pad_trajectories(self, output_teafo_pad_de, lengths_teafo, output_aureg_pad_de, lengths_aureg):
        # unpad the output sequences
        output_teafo_unpad = [out[:len_real] for out, len_real in zip(output_teafo_pad_de, lengths_teafo)]
        output_aureg_unpad = [out[:len_real] for out, len_real in zip(output_aureg_pad_de, lengths_aureg)]
        # merge two sequences
        predicted_seq = self.concat_output_seq(output_teafo_unpad, output_aureg_unpad)
        return predicted_seq

    # def compute_total_gradient_norm(self, models):
    #     total_norm = 0
    #     for model in models:  # Iterate through encoder, LSTM, decoder
    #         for param in model.parameters():
    #             if param.grad is not None:
    #                 param_norm = param.grad.data.norm(2)  # L2 norm
    #                 total_norm += param_norm.item() ** 2
    #     return total_norm ** 0.5  # Final L2 norm

    def compute_total_gradient_norm(self, models):
        """
        Tính tổng norm L2 của gradient từ nhiều mô hình (models).
        Args:
            models (list): Danh sách các mô hình (encoder, LSTM, decoder, etc.).
            
        Returns:
            float: Tổng gradient norm L2, hoặc inf/nan nếu phát hiện bất thường.
        """
        # Lấy thiết bị từ mô hình đầu tiên
        device = next(models[0].parameters()).device
        total_norm = 0.0  # Tích lũy tổng bình phương gradient norm
        
        for model in models:
            for param in model.parameters():
                if param.grad is not None:  # Kiểm tra nếu tham số có gradient
                    # Tính norm L2 của gradient
                    param_norm = param.grad.norm(2)  
                    # Kiểm tra giá trị inf hoặc nan
                    if torch.isinf(param_norm) or torch.isnan(param_norm):
                        print(f"Warning: Gradient norm is inf or NaN for parameter {param}")
                        return float('inf') if torch.isinf(param_norm) else float('nan')
                    
                    # Cộng dồn bình phương norm
                    total_norm += param_norm.item() ** 2
        
        # Trả về tổng norm L2 (căn bậc hai tổng bình phương)
        total_norm = total_norm ** 0.5
        return total_norm


    def save_model_config(self, model_dir):
        with open(os.path.join(model_dir, 'model_config.json'), 'w') as f:
            json.dump(self.model_config, f, indent=4)
        self.util_printer.print_green(f'Model config was saved to {model_dir}', background=True)

    def check_normalization_files_exist(self, path):
        """
        Check if the provided path contains a folder named 'normalization' 
        and if that folder contains specific files with a .save extension.

        Args:
            path (str): The root directory path to check.

        Returns:
            dict: A dictionary with the paths of specific .save files if they exist, 
                or None for each file if it does not exist.
        """
        # Construct the path to the 'normalization' folder
        normalization_path = os.path.join(path, 'normalization')

        # Check if 'normalization' folder exists and is a directory
        if not os.path.isdir(normalization_path):
            return None

        # Define the target files to search for
        target_files = [
            'position_scaler.save', 
            'velocity_scaler.save', 
            'acceleration_scaler.save'
        ]

        # Initialize a dictionary to store the results
        file_paths = {file_name: None for file_name in target_files}

        # Search for the target files in the 'normalization' folder
        for file_name in os.listdir(normalization_path):
            if file_name in target_files:
                file_paths[file_name] = os.path.join(normalization_path, file_name)

        # Check if any of the files were found
        if any(file_paths.values()):
            return file_paths

        return None

def main():
    device = torch.device('cuda')
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/nae_core/data/nae_paper_dataset/data_preprocessed/Bottle'
    thrown_object = 'bottle-1loss1-inlen-2-wc1e-4'
    
    checkout_path = None
    wdb_run_id=None   # 't5nlloi0'
    wdb_resume=None   # 'allow'
    enable_wandb = True

    # Training parameters 
    training_params = {
        'num_epochs': 12000,
        'batch_size_train': 512,    
        'save_interval': 10,
        'thrown_object' : thrown_object,
        'train_id': 'ACC-repair',
        'warmup_steps': 25,
        'dropout_rate': 0.0,
        'loss1_weight': 1.0,
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
    data_params = {
        'step_start': 2,
        'step_end': -2,
        'increment': 1,
    }

    wdb_notes = f'lr: {model_params["lr"]}, \
                {training_params["loss2_weight"]}*L2: ,  \
                {training_params["loss2_1_weight"]}*L2_1, \
                warmup {training_params["warmup_steps"]}, \
                dropout: {training_params["dropout_rate"]}, \
                weight_decay: {training_params["weight_decay"]}'

    nae = NAEDynamicLSTM(**model_params, **training_params, **data_params, data_dir=data_dir, device=device)
    # load data
    nae_data_loader = NAEDataLoader()
    data_train, data_val, data_test = nae_data_loader.load_train_val_test_dataset(data_dir)

    # prepare data for training
    input_label_generator = InputLabelGenerator()
    data_train = input_label_generator.generate_input_label_dynamic_seqs(data_train, step_start=data_params['step_start'], step_end=data_params['step_end'], increment=data_params['increment'], shuffle=True)
    data_val = input_label_generator.generate_input_label_dynamic_seqs(data_val, step_start=data_params['step_start'], step_end=data_params['step_end'], increment=data_params['increment'], shuffle=True)
    data_test = input_label_generator.generate_input_label_dynamic_seqs(data_test, step_start=data_params['step_start'], step_end=data_params['step_end'], increment=data_params['increment'], shuffle=True)

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