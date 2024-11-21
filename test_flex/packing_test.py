from torch.nn.utils.rnn import pack_padded_sequence
import torch

# Dữ liệu sau padding
padded_sequences = torch.tensor([
    [[0.5, 0.6, 0.7], [0.8, 0.9, 1.0], [0.0, 0.0, 0.0]],  # seq_len=3, feature_dim=3
    [[0.1, 0.2, 0.3], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]   # seq_len=1, feature_dim=3
])

# Độ dài thực của các chuỗi
lengths = torch.tensor([2, 1])  # Batch 1: seq_len=2, Batch 2: seq_len=1

# Packing
packed_sequences = pack_padded_sequence(padded_sequences, lengths, batch_first=True, enforce_sorted=False)
print(packed_sequences)