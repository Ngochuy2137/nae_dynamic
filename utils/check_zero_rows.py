import torch

def has_all_zero_rows(mask_matrix):
    """
    Kiểm tra xem mask_matrix có hàng nào toàn phần tử 0 hoặc False hay không.

    Args:
        mask_matrix (torch.Tensor): Ma trận mask có kích thước [batch_size, sequence_length].

    Returns:
        bool: True nếu tồn tại ít nhất một hàng toàn 0 hoặc False, ngược lại False.
        list: Danh sách các chỉ số hàng (row indices) có toàn phần tử 0 hoặc False.
    """
    if not isinstance(mask_matrix, torch.Tensor):
        raise ValueError("mask_matrix phải là torch.Tensor")

    # Tính tổng mỗi hàng
    row_sums = torch.sum(mask_matrix, dim=1)

    # Tìm các hàng có tổng bằng 0
    zero_row_indices = torch.where(row_sums == 0)[0].tolist()

    # Kiểm tra nếu có ít nhất một hàng toàn 0
    has_zero_rows = len(zero_row_indices) > 0

    return has_zero_rows, zero_row_indices

# Ví dụ sử dụng
mask_matrix = torch.tensor([
    [1, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
], dtype=torch.float32)

has_zero, zero_indices = has_all_zero_rows(mask_matrix)
print("Has all-zero rows:", has_zero)
print("Indices of all-zero rows:", zero_indices)