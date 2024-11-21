# from NAE.utils import plot_samples
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def look_for_all_npz_files(folder_dir):
    npy_files = []
    for root, dirs, files in os.walk(folder_dir):
        for file in files:
            if file.endswith('.npz'):
                file_path = os.path.join(root, file)
                npy_files.append(file_path)
    print('Found ', len(npy_files), ' npz files in: ', folder_dir)
    return npy_files

def load_dataset_from_folder(data_folder):
    data_files = look_for_all_npz_files(data_folder)
    dataset = {'position': [], 'time_step': []}
    for data_path in data_files:
        traj_i = np.load(data_path, allow_pickle=True)
        dataset['position'].append(traj_i['position'])
        dataset['time_step'].append(traj_i['time_step'])
    dataset['position'] = np.array(dataset['position'], dtype=object)
    dataset['time_step'] = np.array(dataset['time_step'], dtype=object)
    return dataset

def plot_sample(samples, title, swap_y_z=False):
    fig = plt.figure()
    # # Đặt cửa sổ hiển thị toàn màn hình
    # fig.canvas.manager.full_screen_toggle()

    # Đặt kích thước cửa sổ lớn
    fig.set_size_inches(24, 16)  # Tùy chỉnh kích thước theo ý bạn

    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'pink', 'brown', 'gray']
    print('samples: ', len(samples))
    
    for i in range(len(samples)):
        sample = np.array(samples[i])
        print('sample shape: ', sample.shape)
        
        if swap_y_z:
            sample[:, [1, 2]] = sample[:, [2, 1]]  # Swap Y và Z nếu cần
        
        # Vẽ toàn bộ trajectory với ký hiệu 'o' nhưng chỉ giữ lại những điểm không phải 5 điểm cuối
        ax.plot(sample[:-5, 0], sample[:-5, 1], sample[:-5, 2], 
                'o', color=colors[i % len(colors)], alpha=0.5, label='Test ' + str(i+1) + ' Sample Trajectory')
        
        # Thay thế 5 điểm cuối bằng ID của sample
        for j in range(1, 6):  # Lấy 5 điểm cuối từ -5 đến -1
            ax.text(sample[-j, 0], sample[-j, 1], sample[-j, 2], str(i), 
                    color=colors[i % len(colors)], fontsize=10, fontweight='bold')

        # Đặt giới hạn trục cho đồ thị
        max_x = max(sample[:, 0].max(), ax.get_xlim()[1])
        min_x = min(sample[:, 0].min(), ax.get_xlim()[0])
        max_y = max(sample[:, 1].max(), ax.get_ylim()[1])
        min_y = min(sample[:, 1].min(), ax.get_ylim()[0])
        max_z = max(sample[:, 2].max(), ax.get_zlim()[1])
        min_z = min(sample[:, 2].min(), ax.get_zlim()[0])
        ax.set_xlim([min_x, max_x])
        ax.set_ylim([min_y, max_y])
        ax.set_zlim([min_z, max_z])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.legend()
    plt.title('3D samples ' + title)
    plt.show()

# Hàm chia dữ liệu thành các mảng nhỏ với mỗi mảng gồm 10 phần tử
def split_dataset(dataset, chunk_size=10):
    return [dataset[i:i + chunk_size] for i in range(0, len(dataset), chunk_size)]

def main():
    data_folder = "/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/new_nae/isaac_sim_learning/isaacsim-nae/NAE/data/nae_paper_dataset/origin/trimmed_Paige_171"
    dataset = load_dataset_from_folder(data_folder)

    for i in range(len(dataset['time_step'])):
        # check frequency based on time_step
        one_trajectory = dataset['time_step'][i]
        frequency = 1/np.diff(one_trajectory)
        # print('frequency: ', frequency)
        for f in frequency:
            if f <118:
                print('f: ', f)
    input('hello')

    # dataset_split = split_dataset(dataset, chunk_size=5)
    # for dataset_i in dataset_split:
    #     plot_sample(dataset_i, 'data', swap_y_z = True)

    len_dataset = [len(data) for data in dataset]
    print('check length of trajectories')
    print('min: ', min(len_dataset))
    print('max: ', max(len_dataset))

if __name__ == '__main__':
    main()