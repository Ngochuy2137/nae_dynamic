import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch.optim as optim
import random

from python_utils.python_utils.printer import Printer

DEVICE = torch.device('cuda')

class TimeSeriesDataset(Dataset):
    def __init__(self, num_samples=100, max_len=10, hidden_dim=32):
        self.data = []
        for _ in range(num_samples):
            seq_len_in = random.randint(3, max_len)
            seq_len_out = random.randint(1, max_len)

            # print('check seq_len_in, seq_len_out shape: ', seq_len_in, seq_len_out)
            # input()
            input_seq = torch.rand(seq_len_in, hidden_dim)  # 1 feature per timestep
            label_teafo_seq = input_seq.clone()
            label_aureg_seq = torch.rand(seq_len_out, hidden_dim)  # Output sequence
            self.data.append((input_seq, label_teafo_seq, label_aureg_seq))
        
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
        output_teafo_unpad = [] # teacher forcing output sequences

        #----- teacher forcing -----
        output, (hi, ci) = self.lstm(packed_x, (hi, ci))

        # unpack and only keep real steps, not the padding based on lengths_teacher_forcing
        # Unpack dữ liệu packed output
        output_teafo_pad, len_real_teafo = pad_packed_sequence(output, batch_first=True)
        # print('lengths_teacher_forcing: ', lengths_teacher_forcing)
        # print('len_real_teafo: ', len_real_teafo)


        # Slicing để lấy từng chuỗi theo độ dài thực
        batch_indices = torch.arange(output_teafo_pad.size(0))  # [0, 1, 2]
        output_teafo_unpad = [output_teafo_pad[i, :len_real_teafo[i], :] for i in batch_indices]

        
        #----- autoregressive -----
        # 1. Lấy output cuối cùng của teacher forcing
        input_aureg_init = [out[-1] for out in output_teafo_unpad]
        input_aureg_init = torch.stack(input_aureg_init, dim=0).unsqueeze(1)  # (batch_size, 1, hidden_size)
        # print('check input_aureg_init shape: ', input_aureg_init.shape)
        # print('check hi shape: ', hi.shape)
        # print('check ci shape: ', ci.shape)
        # input()

        out_aureg_unpad = self.auto_regressive_loop(input_aureg_init, hi, ci, lengths_autoregressive)
        # Nối 2 chuỗi output lại với nhau (dim=1 là chiều thời gian)

        # print('out_aureg_unpad length: ', len(out_aureg_unpad))
        # print('lengths_teacher_forcing: ', lengths_teacher_forcing)
        # print('lengths_autoregressive: ', lengths_autoregressive)
        # for seq in out_aureg_unpad:
        #     print('     seq shape: ', seq.shape)

        # out_seq = self.concat_output_seq(output_teafo_unpad, out_aureg_unpad)

        # padding out_aureg_unpad
        out_aureg_pad = pad_sequence(out_aureg_unpad, batch_first=True) 
        # print('check output_teafo_pad shape: ', output_teafo_pad.shape)
        # print('check out_aureg_pad shape: ', out_aureg_pad.shape)
        # input()
        return output_teafo_pad, out_aureg_pad
    
    def concat_output_seq(self, output_teafo_unpad, out_seq_aureg):
        # Nối 2 chuỗi output lại với nhau (dim=1 là chiều thời gian)
        out_seqs = []
        print('check output_teafo_unpad shape: ', len(output_teafo_unpad))
        print('check out_seq_aureg shape: ', len(out_seq_aureg))
        for seq_teafo, seq_aureg in zip(output_teafo_unpad, out_seq_aureg):
            # extend the teacher forcing sequence with the autoregressive sequence (tensors)
            # seq_teafo = torch.cat([seq_teafo, seq_aureg], dim=1)
            print('check seq_teafo shape: ', seq_teafo.shape)
            print('check seq_aureg shape: ', seq_aureg.shape)
            # merge two sequences
            seq_teafo = torch.cat([seq_teafo, seq_aureg], dim=0)
            print('len seq_teafo: ', seq_teafo.shape)
            input()
            out_seqs.append(seq_teafo)
        return out_seqs
        
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

        # # Danh sách lưu trữ các bước output
        # output_seq = []
        # Danh sách lưu trữ các bước output
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

            # print('check temp_output shape: ', temp_output.shape)
            # print('check current_mask shape: ', current_mask.shape)
            # print('check lstm_input shape: ', lstm_input.shape)
            # print('check output shape: ', output.shape)
            # input()
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
    def __init__(self, input_size, hidden_size, num_layers_lstm, lr, device=torch.device('cuda')):
        self.device = device
        
        self.encoder = Encoder(input_size, hidden_size).to(device)
        self.vls_lstm = VLSLSTM(input_size, hidden_size, num_layers_lstm).to(device)
        self.decoder = Decoder(hidden_size, input_size).to(device)

        # # Initialize model, loss, optimizer
        self.criterion = nn.MSELoss(reduction='none').to(device)  # NOTE: Reduction 'none' to apply masking, default is 'mean'
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.vls_lstm.parameters()) + list(self.decoder.parameters()), lr=lr)

        self.util_printer = Printer()


    def collate_pad_fn(self, batch):
        inputs, labels_teafo, labels_aureg = zip(*batch)
        # labels_teafo = [label[:len(input)] for input, label in zip(inputs, labels)]
        # labels_aureg = [label[len(input):] for input, label in zip(inputs, labels)]

        lengths_in = [len(seq) for seq in inputs]
        lengths_teafo = [len(seq) for seq in labels_teafo]

        for lin, lte in zip(lengths_in, lengths_teafo):
            if lin != lte:
                print('check lengths_in, lengths_teafo: ', lin, lte)
                return
        lengths_aureg = [len(seq) for seq in labels_aureg]
        
        # Padding sequences to have same length in batch
        inputs_pad = pad_sequence(inputs, batch_first=True)  # Shape: (batch_size, max_seq_len_in, 1)
        # labels_pad = pad_sequence(labels, batch_first=True)  # Shape: (batch_size, max_seq_len_out, 1)
        labels_teafo_pad = pad_sequence(labels_teafo, batch_first=True)
        labels_aureg_pad = pad_sequence(labels_aureg, batch_first=True)
        
        return inputs_pad, labels_teafo_pad, labels_aureg_pad, lengths_teafo, lengths_aureg

    def train(self, num_epochs, dataset):
        self.encoder.train()
        self.vls_lstm.train()
        self.decoder.train()

        dataloader = DataLoader(dataset, batch_size=4, collate_fn=lambda x: self.collate_pad_fn(x), shuffle=True)
        for epoch in range(num_epochs):
            loss_epoch_log = 0
            for batch in dataloader:
                # print('-----')
                inputs, labels_teafo_pad, labels_aureg_pad, lengths_teafo, lengths_aureg = batch
                inputs = inputs.to(DEVICE)
                labels_teafo_pad = labels_teafo_pad.to(DEVICE)
                labels_aureg_pad = labels_aureg_pad.to(DEVICE)
                lengths_teafo, lengths_aureg = torch.tensor(lengths_teafo, dtype=torch.int64), torch.tensor(lengths_aureg, dtype=torch.int64)

                output_teafo_pad, out_aureg_pad = self.vls_lstm(inputs.float(), lengths_teafo, lengths_aureg)



                # Tạo mask dựa trên chiều dài thực
                mask_teafo = torch.arange(max(lengths_teafo)).expand(len(lengths_teafo), max(lengths_teafo)) < lengths_teafo.unsqueeze(1) # shape: (batch_size, max_seq_len_out)
                mask_aureg = torch.arange(max(lengths_aureg)).expand(len(lengths_aureg), max(lengths_aureg)) < lengths_aureg.unsqueeze(1) # shape: (batch_size, max_seq_len_out)
                # mask = mask.to(labels.device).unsqueeze(2).expand(-1, -1, hidden_dim)  # Mở rộng theo feature

                
                # print('LOSS 1: ')
                # print(f'        lengths_teafo: {lengths_teafo}')
                # print(f'        output_teafo_pad shape: {output_teafo_pad.shape}')
                loss_1 = self.criterion(output_teafo_pad, labels_teafo_pad).sum(dim=-1)  # Shape: (batch_size, max_seq_len_out)


                # print('LOSS 2: ')
                # print(f'        lengths_aureg: {lengths_aureg}')
                # print(f'        out_aureg_pad shape: {out_aureg_pad.shape}')
                loss_2 = self.criterion(out_aureg_pad, labels_aureg_pad).sum(dim=-1)

                # change device of mask
                mask_teafo = mask_teafo.to(loss_1.device)
                loss_1_mask = loss_1 * mask_teafo  # Masked loss
                loss_1_mean = loss_1_mask.sum() / mask_teafo.sum()

                mask_aureg = mask_aureg.to(loss_2.device)
                loss_2_mask = loss_2 * mask_aureg
                loss_2_mean = loss_2_mask.sum() / mask_aureg.sum()

                loss_mean = loss_1_mean + loss_2_mean

                # Backward pass
                self.optimizer.zero_grad()
                loss_mean.backward()
                self.optimizer.step()

                loss_epoch_log += loss_mean.item()

                # print loss
            self.util_printer.print_green(f"\nEpoch {epoch+1}, loss: {loss_epoch_log:.4f}")

def main():
    device = torch.device('cuda')
    # 1. Dataset and DataLoader
    dataset = TimeSeriesDataset(num_samples=100, max_len=15)

    # 2. Initialize model, loss, optimizer
    model = NAEDynamicLSTM(input_size=32, hidden_size=32, num_layers_lstm=2, lr=0.001, device=device)

    # 3. Training loop
    num_epochs = 10
    model.train(num_epochs, dataset)

if __name__ == '__main__':
    main()