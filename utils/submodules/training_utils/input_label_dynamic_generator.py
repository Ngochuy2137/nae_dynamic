import numpy as np
import torch

class InputLabelDynamicGenerator:
    def __init__(self) -> None:
        pass
    
    def create_dynamic_input_label_pairs(self, sequence, step_begin, increment):
        """
        Tạo nhiều cặp input-label từ một chuỗi, với chiều dài input tăng dần.
        Mỗi input được dùng để dự đoán toàn bộ chuỗi đầu ra.
        
        Args:
            sequence (np array): Chuỗi dữ liệu gốc.
        
        Returns:
            list of tuple: Danh sách các cặp (input, label).
        """
        pairs = []
        for i in range(step_begin, len(sequence), increment):
            input_data = sequence[:i]  # Input tăng dần
            label_data = sequence[:]   # Label luôn là toàn bộ chuỗi
            pairs.append((torch.tensor(input_data), torch.tensor(label_data)))
        return pairs