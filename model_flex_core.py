import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch.optim as optim
import random

HIDEN_DIM = 32
DEVICE = torch.device('cuda')

class TimeSeriesDataset(Dataset):
    def __init__(self, num_samples=100, max_len=10):
        self.data = []
        for _ in range(num_samples):
            seq_len_in = random.randint(3, max_len)
            seq_len_out = random.randint(3, max_len)

            # print('check seq_len_in, seq_len_out shape: ', seq_len_in, seq_len_out)
            # input()
            input_seq = torch.rand(seq_len_in, HIDEN_DIM)  # 1 feature per timestep
            label_seq = torch.rand(seq_len_out, HIDEN_DIM)  # Output sequence
            self.data.append((input_seq, label_seq))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
class MultiStepLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MultiStepLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x, lengths, target_length):

        # Khởi tạo hidden state
        h_n, c_n = None, None
        outputs = []
        # print('lengths: ', lengths)
        # input()
        # Bước đầu tiên: Dùng input ban đầu
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, (h_n, c_n) = self.lstm(packed_x)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        # Lấy dự đoán đầu tiên
        outputs.append(lstm_out[:, -1, :].unsqueeze(1))  # Lấy kết quả cuối cùng

        # Các bước tiếp theo
        for _ in range(target_length - 1):
            out, (h_n, c_n) = self.lstm(outputs[-1], (h_n, c_n))
            outputs.append(out)

        # Ghép kết quả dự đoán
        outputs = torch.cat(outputs, dim=1)  # (batch_size, target_length, output_dim)
        return outputs

    

#------------------------- TRAINING -------------------------
# 1. Dataset and DataLoader
dataset = TimeSeriesDataset(num_samples=100, max_len=15)
dataloader = DataLoader(dataset, batch_size=4, collate_fn=lambda x: collate_pad_fn(x), shuffle=True)

def collate_pad_fn(batch):
    inputs, labels = zip(*batch)
    lengths_in = [len(seq) for seq in inputs]
    lengths_out = [len(seq) for seq in labels]
    
    # Padding sequences to have same length in batch
    padded_inputs = pad_sequence(inputs, batch_first=True)  # Shape: (batch_size, max_seq_len_in, 1)
    padded_labels = pad_sequence(labels, batch_first=True)  # Shape: (batch_size, max_seq_len_out, 1)
    
    return padded_inputs, padded_labels, lengths_in, lengths_out


# 2. Initialize model, loss, optimizer
model = MultiStepLSTM(input_dim=HIDEN_DIM, hidden_dim=HIDEN_DIM).to(DEVICE)
criterion = nn.MSELoss(reduction='none').to(DEVICE)  # NOTE: Reduction 'none' to apply masking, default is 'mean'
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, labels, lengths_in, lengths_out = batch
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        lengths_in, lengths_out = torch.tensor(lengths_in, dtype=torch.int64), torch.tensor(lengths_out, dtype=torch.int64)

        # Lấy chiều dài lớn nhất của label trong batch
        max_label_length = lengths_out.max()

        # Forward pass
        # NOTE: Tất cả cặp input-label trong batch đều được feed vào model lstm và được lặp lại target_length lần giống nhau 
        # bất kể label có độ dài khác nhau vì ta đang xử lý theo batch nên không thể xử lý riêng lẻ từng cặp input-label
        outputs = model(inputs.float(), lengths_in, target_length=max_label_length)

        # Tạo mask dựa trên chiều dài thực
        mask = torch.arange(max(lengths_out)).expand(len(lengths_out), max(lengths_out)) < torch.tensor(lengths_out).unsqueeze(1) # shape: (batch_size, max_seq_len_out)
        # mask = mask.to(labels.device).unsqueeze(2).expand(-1, -1, HIDEN_DIM)  # Mở rộng theo feature

        loss_raw = criterion(outputs, labels).sum(dim=-1)  # Shape: (batch_size, max_seq_len_out)

        # change device of mask
        mask = mask.to(loss_raw.device)
        loss_mask = loss_raw * mask  # Masked loss
        loss_mean = loss_mask.sum() / mask.sum()

        # Backward pass
        optimizer.zero_grad()
        loss_mean.backward()
        optimizer.step()

        # print loss
        print(f"Epoch {epoch+1}, loss: {loss_mean.item():.4f}")
