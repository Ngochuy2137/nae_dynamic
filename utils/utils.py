import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from .submodules.plotter import RoCatDataPlotter
from torch.utils.data import DataLoader, TensorDataset

class NAE_Utils:
    def __init__(self):
        self.plotter = RoCatDataPlotter()
        
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

    def save_model_info(self, data_dir, model_dir, data_num, num_epochs, batch_size, seq_length, prediction_steps, start_t, training_t, wandb_run_url,loss_all_data, loss_graph_image_path):        
        readme_file = os.path.join(model_dir, 'README.md')
        final_loss = loss_all_data[-1][-1]  # Final total loss
        
        with open(readme_file, 'w') as f:
            f.write("# Model Information\n")

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
            f.write(f"| **Sequence Length**      | {seq_length} |\n")
            f.write(f"| **Prediction Steps**     | {prediction_steps} |\n")
            f.write(f"| **Final Loss**           | {final_loss:.9f} |\n")
            f.write(f"| **Wandb Run URL**        | {wandb_run_url} |\n")

            f.write("\n## Losses\n")
            f.write("| Epoch | Loss1 | Loss2 | Loss3 | Total Loss |\n")
            f.write("|-------|-------|-------|-------|------------|\n")
            loss1 = loss_all_data[0][0]
            loss2 = loss_all_data[1][0]
            loss3 = loss_all_data[2][0]
            total_loss = loss_all_data[3][0]
            f.write(f"| {0} | {loss1:.9f} | {loss2:.9f} | {loss3:.9f} | {total_loss:.9f} |\n")
            loss1 = loss_all_data[0][-1]
            loss2 = loss_all_data[1][-1]
            loss3 = loss_all_data[2][-1]
            total_loss = loss_all_data[3][-1]
            f.write(f"| {len(loss_all_data[0])-1} | {loss1:.9f} | {loss2:.9f} | {loss3:.9f} | {total_loss:.9f} |\n")
            f.write("\n## Training Loss Graph\n")
            relative_image_path = os.path.relpath(loss_graph_image_path, model_dir)
            f.write(f"![Training Loss]({relative_image_path})\n")
    
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
    

    def score_all_predictions(self, predictions, labels, future_pred_steps, capture_thres=0.1):
        # Tính toán MSE cho tất cả các phần tử
        mse_all = np.mean((predictions - labels) ** 2, axis=(1, 2))
        
        # Tính toán MSE cho các thành phần x, y, z
        mse_xyz = np.mean((predictions[:, :, :3] - labels[:, :, :3]) ** 2, axis=(1, 2))
        
        # Tính toán Average Displacement Error (ADE)
        displacement_err_list = np.linalg.norm(predictions[:, :, :3] - labels[:, :, :3], axis=2)
        ade = np.mean(displacement_err_list, axis=1)
        
        # Tính toán Norm Average Displacement Error (NADE)
        # sub_length = np.linalg.norm(one_label[1:] - one_label[:-1], axis=1)
        # total_length = np.sum(sub_length)
        # nade = ade/total_length

        sub_length = np.linalg.norm(labels[:, 1:, :3] - labels[:, :-1, :3], axis=2)
        # sub_length = np.linalg.norm(labels[:, 1:] - labels[:, :-1], axis=2)
        total_length = np.sum(sub_length, axis=1)
        nade = ade / total_length

        
        # NADE for future prediction
        displacement_err_list_future = np.linalg.norm(predictions[:, -future_pred_steps:, :3] - labels[:, -future_pred_steps:, :3], axis=2)
        ade_future = np.mean(displacement_err_list_future, axis=1)
        total_length_future = np.sum(sub_length[:, -future_pred_steps:], axis=1)
        nade_future = ade_future / total_length_future
        
        # Tính toán khoảng cách cuối cùng
        final_step_err = displacement_err_list[:, -1]




        # Calculate mean values of all predictions
        mean_mse_all = np.mean(mse_all)
        mean_mse_xyz = np.mean(mse_xyz)
        mean_ade = np.mean(ade)

        mean_nade = np.mean(nade)
        # Tính phương sai (Variance)
        var_nade = np.var(nade)

        mean_final_step_err = np.mean(final_step_err)
        # Tính phương sai (Variance)
        var_fe = np.var(final_step_err)

        mean_nade_future = np.mean(nade_future)

        # Calculate capture success rate
        count_less_than_threshold = np.sum(final_step_err <= capture_thres)
        capture_success_rate = (count_less_than_threshold / final_step_err.size) * 100
        # input('\nTODO: check final_step_err.size and capture_success_rate size')
        # print('final_step_err: ', final_step_err)
        # print(f'final_step_err.size: {final_step_err.size}, capture_success_rate: {capture_success_rate}')
        
        return mean_mse_all, mean_mse_xyz, mean_ade, (mean_nade,var_nade), (mean_final_step_err, var_fe), mean_nade_future, capture_success_rate
    
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