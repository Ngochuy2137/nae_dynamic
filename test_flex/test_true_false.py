import torch
import numpy as np

# simulate padding, masking sequences
real_lengths = torch.tensor([1, 2, 5])

# create a padded sequence matrix with shape (3, 5) based on real_lengths
seq1 = torch.tensor([1, 0, 0, 0, 0])
seq2 = torch.tensor([1, 2, 0, 0, 0])
seq3 = torch.tensor([1, 2, 3, 4, 5])
seq_padded = torch.stack([seq1, seq2, seq3], dim=0)

# create mask matrix
max_len = seq_padded.size(1)


print(torch.arange(max_len).unsqueeze(0))
print(real_lengths.unsqueeze(1))
input()
'''
seq_padded.size(0) is batch size
seq_padded.size(1) is sequence length (padded)
seq_padded.size(2) is feature size
'''
mask = torch.arange(max_len).unsqueeze(0) < real_lengths.unsqueeze(1)

print(mask)
'''
tensor([[ True, False, False, False, False],
        [ True,  True, False, False, False],
        [ True,  True,  True,  True,  True]])
'''

# Iterate over each time step
for t in range(max_len):
    print(f"Time step {t}")
    current_mask = mask[:, t]
    print(f"    current_mask: {current_mask}")
    seq_padded_t = seq_padded[:, t]
    print(f"    seq_padded_t: {seq_padded_t}")
    seq_padded_t_masked = seq_padded_t[current_mask]
    print(f"    seq_padded_t_masked: {seq_padded_t_masked}")

    input()

