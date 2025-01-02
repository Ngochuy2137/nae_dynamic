import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from .submodules.plotter import RoCatDataPlotter
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from python_utils.printer import Printer

global_util_printer = Printer()

class NAE_Utils:
    def __init__(self):
        self.plotter = RoCatDataPlotter()
    
    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
    def generate_one_parabol(self, min_vel, max_vel, min_theta, max_theta, min_phi, max_phi, delta_t, time_steps, g=9.81):
        v0 = np.random.uniform(min_vel, max_vel)
        theta = np.random.uniform(np.radians(min_theta), np.radians(max_theta))
        phi = np.random.uniform(np.radians(min_phi), np.radians(max_phi))

        t = np.linspace(0, delta_t * time_steps, time_steps)
        x = v0 * np.cos(theta) * np.cos(phi) * t
        y = v0 * np.cos(theta) * np.sin(phi) * t
        z = -0.5 * g * (t ** 2) + v0 * np.sin(theta) * t
        vx = np.full_like(x, v0 * np.cos(theta) * np.cos(phi))
        vy = np.full_like(x, v0 * np.cos(theta) * np.sin(phi))
        vz = v0 * np.sin(theta) - g * t
        ax = np.full_like(x, 0.0)
        ay = np.full_like(x, 0.0)
        az = np.full_like(x, -g)
        return np.stack((x, y, z, vx, vy, vz, ax, ay, az), axis=1)

    def generate_parabolic_motion_data(self, min_vel, max_vel, min_theta, max_theta, min_phi, max_phi, delta_t, time_steps, g, num_samples):
        # Dataset
        data = []
        for _ in range(num_samples):
            trajectory = self.generate_one_parabol(min_vel, max_vel, min_theta, max_theta, min_phi, max_phi, delta_t, time_steps, g)
            data.append(trajectory)
        data = np.array(data)
        return data

    def look_for_file(self, base_dir, file_name):
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file == file_name:
                    return os.path.join(root, file)
        return None

    def save_model_info(self, data_dir, model_dir, data_num, num_epochs, batch_size, start_t, training_t, loss_all_data, wandb_run_url=''): 
        readme_file = os.path.join(model_dir, 'README.md')
        final_loss = loss_all_data[-1][-1]  # Final total loss
        
        with open(readme_file, 'w') as f:
            f.write("# Model Information\n")
            f.write("# NAE DYNAMIC\n")
            f.write("\n## Params\n")
            f.write("| Parameter                | Value |\n")
            f.write("|--------------------------|-------|\n")
            f.write(f"| **Start Time**           | {start_t} |\n")
            f.write(f"| **Training period (mins)**| {(training_t):.2f} |\n")
            f.write(f"| **Training data dir**    | {data_dir} |\n")
            f.write(f"| **Saved model dir**      | {model_dir} |\n")
            f.write(f"| **Training data size**   | {data_num} |\n")
            f.write(f"| **Number of Epochs**     | {num_epochs} |\n")
            f.write(f"| **Batch Size**           | {batch_size} |\n")
            f.write(f"| **Final Loss**           | {final_loss:.9f} |\n")
            f.write(f"| **Wandb Run URL**        | {wandb_run_url} |\n")

            f.write("\n## Losses\n")
            f.write("| Epoch | Loss1 | Loss2 | Loss3 | Total Loss |\n")
            f.write("|-------|-------|-------|-------|------------|\n")
            loss1 = loss_all_data[0][0]
            loss2 = loss_all_data[0][1]
            loss3 = loss_all_data[0][2]
            total_loss = loss_all_data[0][3]
            f.write(f"| {0} | {loss1:.9f} | {loss2:.9f} | {loss3:.9f} | {total_loss:.9f} |\n")
            loss1 = loss_all_data[-1][0]
            loss2 = loss_all_data[-1][1]
            loss3 = loss_all_data[-1][2]
            total_loss = loss_all_data[-1][3]
            f.write(f"| {len(loss_all_data)} | {loss1:.9f} | {loss2:.9f} | {loss3:.9f} | {total_loss:.9f} |\n")
    
    def save_loss(self, loss_all_data, model_dir):
        fig, ax = plt.subplots()
        # Plot loss data
        ax.plot(loss_all_data[0], label='Loss 1')
        ax.plot(loss_all_data[1], label='Loss 2')
        ax.plot(loss_all_data[2], label='Loss 3')
        ax.plot(loss_all_data[3], label='Loss total')

        # Set labels and title
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        
        # Add legend
        ax.legend()
        
        # Set y-axis ticks
        max_loss = max(max(loss_all_data[3]), 500000)  # Ensure that the max value on the y-axis covers the range of your data
        yticks = range(0, int(max_loss) + 10000, 10000)  # Add ticks every 50000 units
        ax.set_yticks(yticks)
        ax.tick_params(axis='both', which='major', labelsize=5)

        # Save the plot
        loss_graph_path = os.path.join(model_dir, 'training_loss.png')
        plt.savefig(loss_graph_path, dpi=300)
        plt.close()  # Đóng plot để giải phóng bộ nhớ
        return loss_graph_path
    

    def score_all_predictions(self, output_teafo_pad_de, labels_teafo_pad, lengths_teafo, mask_teafo,
                                    output_aureg_pad_de, labels_aureg_pad, lengths_aureg, mask_aureg,
                                    capture_thres=[0.01, 0.05, 0.1], debug=False):

        # mask_reconstruction = torch.arange(max(lengths_reconstruction)).expand(len(lengths_reconstruction), max(lengths_reconstruction)) < lengths_reconstruction.unsqueeze(1) # shape: (batch_size, max_seq_len_out)
        # mask_reconstruction = mask_reconstruction.to(loss_3.device)
        
        # calculate mask based on lengths_teafo and lengths_aureg
        # mask_teafo = torch.arange(max(lengths_teafo)).expand(len(lengths_teafo), max(lengths_teafo)) < lengths_teafo.unsqueeze(1) # shape: (batch_size, max_seq_len_out)
        # mask_teafo = mask_teafo.to(output_teafo_pad_de.device)
        # mask_aureg = torch.arange(max(lengths_aureg)).expand(len(lengths_aureg), max(lengths_aureg)) < lengths_aureg.unsqueeze(1)
        # mask_aureg = mask_aureg.to(output_aureg_pad_de.device)
        mask_combined = torch.cat([mask_teafo, mask_aureg], dim=1)

        if debug:
            # check matrix mask_combined is correctly created
            for i, m_cb, m_tf, m_ar in zip(range(len(mask_combined)), mask_combined, mask_teafo, mask_aureg):
                # print(f'Row {i}: -> check {m_cb} = {m_tf} + {m_ar}')
                if torch.sum(m_cb) != torch.sum(m_tf) + torch.sum(m_ar):
                    print(f'Error in mask_combined at row {i}')
                    print(f'mask_combined: {m_cb}')
                    print(f'mask_teafo: {m_tf}')
                    print(f'mask_aureg: {m_ar}')
                    global_util_printer.print_red('Error in mask_combined')
                    raise ValueError('Error in mask_combined')


        output_combined = torch.cat([output_teafo_pad_de, output_aureg_pad_de], dim=1)
        labels_combined = torch.cat([labels_teafo_pad, labels_aureg_pad], dim = 1)
        
        # # Tính toán MSE cho tất cả các phần tử
        # mse_all = np.mean((predictions - labels) ** 2, axis=(1, 2))
        
        # # Tính toán MSE cho các thành phần x, y, z
        # mse_xyz = np.mean((predictions[:, :, :3] - labels[:, :, :3]) ** 2, axis=(1, 2))
        
        
        ## =================== ADE Calculation ===================
        # Tính toán Average Displacement Error (ADE) cho tất cả các phần tử, kết hợp với mask

        #       This is for entire prediction
        ade_entire = self.ade_masked_calculation(output_combined, labels_combined, mask_combined)
        #       This is for future prediction
        ade_future = self.ade_masked_calculation(output_aureg_pad_de, labels_aureg_pad, mask_aureg)
        #       This is for past prediction
        ade_past = self.ade_masked_calculation(output_teafo_pad_de, labels_teafo_pad, mask_teafo)
        
        ## =================== NADE Calculation ===================
        ## (Norm Average Displacement Error)
        #       This is for entire prediction 

        nade_entire = self.nade_masked_calculation(ade_entire, labels_combined, mask_combined)
        #       This is for future prediction
        nade_future = self.nade_masked_calculation(ade_future, labels_aureg_pad, mask_aureg)
        #       This is for past prediction
        nade_past = self.nade_masked_calculation(ade_past, labels_teafo_pad, mask_teafo)
        
        ## =================== Final point Calculation ===================
        #       Calculate final step error
        #       Find the final valid step of each trajectory in batch based on mask
        final_step_err, converge_to_final_point_trending = self.final_step_prediction_error(output_combined, labels_combined, mask_combined, mask_aureg)

        ## =================== Synthesis all results ===================
        # Calculate mean values of all predictions
        # mean_mse_all = np.mean(mse_all)
        # mean_mse_xyz = np.mean(mse_xyz)
        mean_ade_entire = np.mean(ade_entire.cpu().numpy())
        std_ade_entire = np.std(ade_entire.cpu().numpy())

        mean_ade_future = np.mean(ade_future.cpu().numpy())
        std_ade_future = np.std(ade_future.cpu().numpy())

        mean_ade_past = np.mean(ade_past.cpu().numpy())
        std_ade_past = np.std(ade_past.cpu().numpy())

        mean_nade_entire = np.mean(nade_entire.cpu().numpy())
        std_nade_entire = np.std(nade_entire.cpu().numpy())

        mean_nade_future = np.mean(nade_future.cpu().numpy())
        std_nade_future = np.std(nade_future.cpu().numpy())

        mean_nade_past = np.mean(nade_past.cpu().numpy())
        std_nade_past = np.std(nade_past.cpu().numpy())

        mean_fe = np.mean(final_step_err.cpu().numpy()) # final step error
        std_fe = np.std(final_step_err.cpu().numpy())


        # Calculate capture success rate
        capture_success_rates = []
        for cap_thr in capture_thres:
            success_rate_matrix = (final_step_err <= cap_thr).cpu().numpy()
            mean_csr = np.mean(success_rate_matrix)
            # cal standard deviation
            std_csr = np.std(success_rate_matrix)
            capture_success_rates.append((cap_thr, mean_csr, std_csr))
        
        return  (mean_ade_entire,   std_ade_entire), \
                (mean_ade_future,   std_ade_future), \
                (mean_ade_past,     std_ade_past), \
                (mean_nade_entire,  std_nade_entire), \
                (mean_nade_future,  std_nade_future), \
                (mean_nade_past,    std_nade_past), \
                (mean_fe, std_fe, converge_to_final_point_trending), \
                capture_success_rates
    
    
    def ade_masked_calculation(self, predictions, labels, mask_matrix):
        displacement_err_list = torch.norm(predictions - labels, dim=2)*mask_matrix
        ade = torch.sum(displacement_err_list, dim=1)
        # calculate total valide data point of each row in mask_matrix
        total_length = torch.sum(mask_matrix, dim=1)
        # print('check mask_matrix shape: ', mask_matrix.shape)
        # # check if all elements of total_length are the same
        # print('check total_length same: ', torch.all(total_length == total_length[0]))
        # input()
        # print('total_length: ', total_length)
        # input()
        # # filter unique elements in total_length
        # total_length_unique = torch.unique(total_length)
        # print('total_length_unique: ', total_length_unique)
        # input()
        
        ade = ade/total_length  # mean
        return ade

    def nade_masked_calculation(self, ade, labels, mask_matrix):
        # Calculate accumulated displacement (step by step) for each trajectory in batch based on mask
        sub_length = torch.norm(labels[:, 1:, :3] - labels[:, :-1, :3], dim=2)
        mask_combined_valid = mask_matrix[:, 1:] * mask_matrix[:, :-1]
        sub_length_valid = sub_length * mask_combined_valid
        accumulated_length = torch.sum(sub_length_valid, dim=1)    
        nade = ade / accumulated_length
        # check if any row of mask_matrix is all False elements
        matrix_check = torch.sum(mask_matrix, dim=1)
        for m in matrix_check:
            if m == 0:
                print('Found a row with all False elements 1')
                input()
        
        matrix_valid_check = torch.sum(mask_combined_valid, dim=1)
        for i, m in enumerate(matrix_valid_check):
            if m == 0:
                print('Found a row with all False elements 2')
                print(mask_matrix[i])
                print(mask_matrix[i, 1:])
                print(mask_matrix[i, :-1])
                input()
        
        return nade
    
    def final_step_prediction_error(self, predictions, labels, mask_combined, mask_aureg):
        batch_size = mask_combined.size(0)
        step_size = mask_combined.size(1)
        # Tìm step cuối cùng
        reverse_mask = torch.flip(mask_combined, dims=[1]).int()
        last_valid_step = step_size - 1 - torch.argmax(reverse_mask, dim=1)
        # 3. Tính L2 error
        batch_indices = torch.arange(batch_size)
        final_prediction = predictions[batch_indices, last_valid_step, :3]  # [batch_size, dim]
        final_label = labels[batch_indices, last_valid_step, :3]  # [batch_size, dim]

        l2_error = torch.norm(final_prediction - final_label, dim=1)  # [batch_size]
        # calculate sum of each row in mask_aureg
        length_left = torch.sum(mask_aureg, dim=1)

        # calculate penalty for variance of length_left
        converge_to_final_point_trending = self.calculate_penalty_metric(l2_error, length_left)

        return l2_error, converge_to_final_point_trending

    def calculate_penalty_metric(self, errors, steps):
        """
        Calculate penalty-based metric to evaluate errors based on steps using PyTorch.

        Args:
        - errors (torch.Tensor): Tensor of errors for each prediction step.
        - steps (torch.Tensor): Tensor of step indices corresponding to errors.
        - device (str): Device to perform calculations ("cuda" for GPU, "cpu" for CPU).

        Returns:
        - metric (float): Normalized penalty-based metric in [0, 1].
        """
        # check if errors and steps have the same length
        if errors.size(0) != steps.size(0):
            raise ValueError("errors and steps must have the same length.")
        # Sắp xếp errors và steps theo thứ tự tăng dần của steps
        sorted_indices = torch.argsort(steps)
        steps = steps[sorted_indices]
        errors = errors[sorted_indices]

        # Tính trọng số ngược tỉ lệ với step
        max_step = steps.max()
        weights = (max_step - steps + 1) / max_step  # Trọng số
        # # weights = 5/(steps + 1)  # Trọng số
        # weights = torch.exp(-50 * steps)


        # Tính penalty
        penalties = weights * errors
        total_penalty = penalties.sum()

        # Chuẩn hóa penalty
        max_possible_penalty = weights.sum() * errors.max()
        metric = 1 - total_penalty / max_possible_penalty

        # Trả kết quả về CPU dưới dạng float
        return metric.item()

    def prepare_data_loaders(self, data_train, data_val, batch_size_train, batch_size_val):
        #   prepare training data
        x_train, y_train = data_train
        x_train = torch.tensor(x_train, dtype=torch.float32)    # no need to convert to np array in advance
        y_train = torch.tensor(y_train, dtype=torch.float32)
        tr_dataset = TensorDataset(x_train, y_train)
        #   prepare validation data
        x_val, y_val = data_val
        x_val = torch.tensor(x_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)
        val_dataset = TensorDataset(x_val, y_val)

        dl_train = DataLoader(tr_dataset, batch_size=batch_size_train, shuffle=True)
        dl_val = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True)
        return dl_train, dl_val

    '''
    check:
        if input data and label data have the same length

        if len of each input in data_train = input_len (t_value)
        if len of each label in data_train = input_len + future_pred_len (t_value + k_value)

        if len of each input in data_val = input_len (t_value)
        if len of each label in data_val = input_len + future_pred_len (t_value + k_value)
    '''
    def proper_data_check(self, data_train:tuple, data_val:tuple, t_value, k_value):
        # check training data
        x_train, y_train = data_train
        if len(x_train) != len(y_train):
            return False
        for one_x_train, one_y_train in zip(x_train, y_train):
            if len(one_x_train) != t_value or len(one_y_train) != t_value + k_value:
                return False
            
        # check validation data
        x_val, y_val = data_val
        if len(x_val) != len(y_val):
            return False
        for one_x_val, one_y_val in zip(x_val, y_val):
            if len(one_x_val) != t_value or len(one_y_val) != t_value + k_value:
                return False
        return True