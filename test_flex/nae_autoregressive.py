import torch


output_seq = []  # Danh sách dùng để lưu trữ các bước đầu ra (output) qua từng bước thời gian.
lstm_input = batch_x_padded[:, 0:1, :]  # Lấy bước đầu tiên từ chuỗi đã padding làm đầu vào ban đầu.
hi =
ci =
# batch_x_padded: Tensor padded, kích thước (batch_size, seq_len, feature_size).
# lstm_input ban đầu có kích thước (batch_size, 1, feature_size), chỉ chứa bước đầu tiên của mỗi chuỗi.

# Tạo mask
lengths = torch.tensor([3, 2, 1])  # Độ dài thực của mỗi chuỗi
max_len = batch_x_padded.size(1)
mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)  # Mask với True cho các bước thực

for t in range(max_len):  # Duyệt qua từng bước thời gian
    # Lấy mask của các chuỗi thực tại bước t
    current_mask = mask[:, t]   # Lấy mask cho bước thời gian t. 
                                # current_mask là tensor Boolean với kích thước (batch_size,).
                                # Ví dụ: current_mask tại t=1:
                                # [ True, True, False] (chuỗi 1 và 2 có dữ liệu thực, chuỗi 3 là padding).
    # Nếu tất cả các chuỗi tại bước t là padding, bỏ qua bước này
    if not current_mask.any():
        break

    # Truyền lstm_input qua LSTM, nhưng chỉ xử lý các chuỗi không bị padding
    lstm_input = lstm_input[current_mask]   # Lọc lstm_input, chỉ giữ lại các chuỗi thực.
                                            # lstm_input ban đầu có kích thước (batch_size, 1, feature_size).
                                            # Sau khi lọc, kích thước sẽ là (num_real_sequences, 1, feature_size),
                                            # trong đó num_real_sequences = current_mask.sum().

    # hi shape: (num_layers, batch_size, hidden_size)
    # ci shape: (num_layers, batch_size, hidden_size)
    # Giữ nguyên shape của hi, ci, chỉ cập nhật cho các chuỗi thực.
    hi_current = hi[:, current_mask, :]     # Lọc trạng thái hidden (hi) chỉ giữ lại các chuỗi thực.
    ci_current = ci[:, current_mask, :]     # Lọc trạng thái cell (ci) chỉ giữ lại các chuỗi thực.
                                    # hi, ci có kích thước ban đầu (num_layers, batch_size, hidden_size).
                                    # Sau khi lọc, kích thước sẽ là (num_layers, num_real_sequences, hidden_size).


    # Truyền vào LSTM
    output, (hi_new, ci_new) = self.lstm(lstm_input, (hi_current, ci_current))
    # Truyền các chuỗi thực qua LSTM.
    # LSTM nhận vào lstm_input và trạng thái hiện tại (hi, ci).
    # Đầu ra:
    # - output: (num_real_sequences, 1, hidden_size).
    # - hi_new, ci_new: Trạng thái hidden và cell mới cho các chuỗi thực.

    hi[:, current_mask, :] = hi_new  # Cập nhật trạng thái hidden cho các chuỗi thực.
    ci[:, current_mask, :] = ci_new  # Cập nhật trạng thái cell cho các chuỗi thực.
    # Trạng thái của các chuỗi thực được cập nhật.
    # Trạng thái của các chuỗi padding giữ nguyên.

    # Lưu output và đảm bảo giữ kích thước batch
    temp_output = torch.zeros(batch_size, 1, hidden_size).to(output.device)
    # Khởi tạo tensor tạm với kích thước (batch_size, 1, hidden_size), toàn giá trị 0.
    # temp_output sẽ chứa đầu ra cho toàn bộ batch (bao gồm cả chuỗi padding).


    temp_output[current_mask] = output
    # Cập nhật temp_output chỉ với đầu ra của các chuỗi thực.
    # Chuỗi padding giữ nguyên giá trị 0.


    output_seq.append(temp_output)
    # Lưu temp_output vào danh sách output_seq.
    # Kích thước của output_seq sau mỗi bước là (batch_size, 1, hidden_size).


    # Cập nhật lstm_input cho bước tiếp theo
    lstm_input = temp_output[:, -1:, :]  # Lấy đầu ra của bước hiện tại làm đầu vào
