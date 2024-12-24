from nae_core.utils.submodules.preprocess_utils.data_raw_reader import RoCatRLLabDataRawReader, RoCatNAEDataRawReader
# from utils.submodules.preprocess_utils.merger import RoCatRLLabDataMerger, RoCatNAEDataMerger
from nae_core.utils.submodules.training_utils.input_label_generator import InputLabelGenerator
from nae_core.utils.submodules.training_utils.data_loader import DataLoader as NAEDataLoader


import numpy as np
import matplotlib.pyplot as plt
import os
from python_utils.plotter import Plotter
from python_utils.printer import Printer
from collections import defaultdict

global_util_printer = Printer()
global_util_plotter = Plotter()

class RoCatDataRawInvestigator:
    def __init__(self,):
        pass

    def analyze_data(self, data_dir):
        nae_data_loader = NAEDataLoader()
        data_train, _, _ = nae_data_loader.load_dataset(data_dir)
        input_label_generator = InputLabelGenerator()
        data_train_splitted = input_label_generator.generate_input_label_dynamic_seqs(data_train, step_start=5, step_end=-3, increment=1, shuffle=True)

        # input_seq, label_teafo_seq, label_aureg_seq, label_reconstruction_seq
        
        length_left_list = [len(label_aureg_seq) for input_seq, label_teafo_seq, label_aureg_seq, label_reconstruction_seq in data_train_splitted]
        # create data to plot
        grouped_data = defaultdict(list)
        # make data to plot based on length_left_list: count appearance of each length
        for length in length_left_list:
            grouped_data[length].append(length)
        # sort data descendingly through key
        grouped_data = dict(sorted(grouped_data.items(), key=lambda item: item[0], reverse=True))
        # get x_values and y_values
        X_values = list(grouped_data.keys())
        y_values = [len(grouped_data[key]) for key in X_values]

        global_util_plotter.plot_line_chart(x_values=X_values, y_values=[y_values], title="Length of left sequence", x_label="Index", y_label="Length", 
                                          save_plot=None, x_tick_distance=5, keep_source_order=True)

# TODO: Sử dụng kế thừa để tạo 2 class con. Class cha chứa các method get_...() để lấy dữ liệu cần thiết từ data_raw
class RoCatRLLabDataRawInvestigator:
    def __init__(self,):
        pass
    
    def plot_start_end_positions(self, trajectories):
        """
        Plot the start and end positions of trajectories.

        Args:
        - trajectories: List of npz-like data, where each item contains 'position' as a key.
                        'position' should be an array of shape (N, 3) representing [x, y, z].

        Returns:
        - None. Displays a 2D plot of start (red) and end (blue) positions.
        """
        start_positions = []
        end_positions = []

        # Extract start and end positions from each trajectory
        for traj in trajectories:
            if 'position' in traj:
                positions = traj['position']
                start_positions.append(positions[0][:2])  # Extract [x, y] for start
                end_positions.append(positions[-1][:2])  # Extract [x, y] for end

        start_positions = np.array(start_positions)
        end_positions = np.array(end_positions)

        # Plot the positions
        plt.figure(figsize=(8, 8))
        plt.scatter(
            start_positions[:, 0], start_positions[:, 1], 
            c='red', label='Start Positions', s=50, edgecolors='black'
        )
        plt.scatter(
            end_positions[:, 0], end_positions[:, 1], 
            c='blue', label='End Positions', s=50, edgecolors='black'
        )

        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("Start and End Positions of Trajectories")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()
    
    def plot_throw_direction_vectors(self, trajectories, data_whose_y_up=False, title_note='', check_starting_point_outlier=False):
        """
        Plot vectors (arrows) from start to end positions of trajectories.

        Args:
        - trajectories: List of npz-like data, where each item contains 'points' as a key.
                        'points' should be an array of shape (N, 3) representing [x, y, z].

        Returns:
        - None. Displays a 2D plot with arrows.
        """
        plt.figure(figsize=(12, 12))
        x_lim_max = -1000
        x_lim_min = 1000
        y_lim_max = -1000
        y_lim_min = 1000

        # Extract and plot vectors
        for idx, traj in enumerate(trajectories):
            if 'points' in traj:
                positions = traj['points']
                if data_whose_y_up:
                    start = [positions[0][0], -positions[0][2], positions[0][1]]  # Start position [x, z]
                    end = [positions[-1][0], -positions[-1][2], positions[0][1]]  # End position [x, z]
                else:
                    start = [positions[0][0], positions[0][1], positions[0][2]]  # Start position [x, z]
                    end = [positions[-1][0], positions[-1][1], positions[0][2]]  # End position [x, z]

                if check_starting_point_outlier:
                    if start[2] >= 1.8:
                        # plot this trajectory
                        plotter = Plotter()
                        plotter.plot_samples([positions], title=f'Outlier trajectory with starting point at [{start[0]:.3f}, {start[1]:.3f}, {start[2]:.3f}]', rotate_data_whose_y_up=True)

                # Draw arrow from start to end
                dx, dy = end[0] - start[0], end[1] - start[1]
                plt.arrow(
                    start[0], start[1], dx, dy, 
                    head_width=0.05, head_length=0.05, fc='green', ec='green', alpha=0.3
                )

                # Mark start position with a red circle
                plt.scatter(
                    start[0], start[1], c='red', s=50, edgecolor='black',
                    label='Start Positions' if idx == 0 else None
                )

                # Update x and y limits
                x_lim_max = max(x_lim_max, start[0], end[0])
                x_lim_min = min(x_lim_min, start[0], end[0])
                y_lim_max = max(y_lim_max, start[1], end[1])
                y_lim_min = min(y_lim_min, start[1], end[1])

        # Set limits for x and y axis
        if x_lim_max - x_lim_min < y_lim_max - y_lim_min:
            x_lim_mid = (x_lim_max + x_lim_min) / 2
            x_lim_min = x_lim_mid - (y_lim_max - y_lim_min) / 2
            x_lim_max = x_lim_mid + (y_lim_max - y_lim_min) / 2
        else:
            y_lim_mid = (y_lim_max + y_lim_min) / 2
            y_lim_min = y_lim_mid - (x_lim_max - x_lim_min) / 2
            y_lim_max = y_lim_mid + (x_lim_max - x_lim_min) / 2
        plt.xlim(x_lim_min - 0.1, x_lim_max + 0.1)
        plt.ylim(y_lim_min - 0.1, y_lim_max + 0.1)

        plt.xlabel("X Position", fontsize=14, labelpad=20)
        plt.ylabel("Y Position", fontsize=14, labelpad=20)
        plt.title(title_note + ' - Throw direction Vectors (Start to End)' + f' - Total data number: {len(trajectories)}', fontsize=20, pad=30)
        plt.grid(True)
        # plt.axis('equal')
        plt.legend(loc='upper left')
        plt.show()

    def get_lengths(self, data_raw):
        lengths = [len(traj['points']) for traj in data_raw]
        return lengths
    
    def get_pick_heights(self, data_raw, data_whose_y_up=False):
        pick_height_list = []
        # Extract max heights for each trajectory
        for traj in data_raw:
            if 'points' in traj:
                positions = traj['points']
                if data_whose_y_up:
                    heights = positions[:, 1]  # Use Y coordinate as height
                else:
                    heights = positions[:, 2]  # Use Z coordinate as height
                max_height = np.max(heights)
                pick_height_list.append(max_height)
        return pick_height_list
    
    def get_throw_direction_angles(self, data_raw, data_whose_y_up=False):
        angles = []
        # Calculate angles for each trajectory
        for traj in data_raw:
            if 'points' in traj:
                positions = traj['points']
                if data_whose_y_up:
                    start = [positions[0][0], -positions[0][2]]  # Start position [x, y]
                    end = [positions[-1][0], -positions[-1][2]]  # End position [x, y]
                else:
                    start = [positions[0][0], positions[0][1]]
                    end = [positions[-1][0], positions[-1][1]]

                # Compute vector and angle in degrees
                dx, dy = end[0] - start[0], end[1] - start[1]
                angle = np.arctan2(dy, dx) * 180 / np.pi  # Convert radians to degrees
                if angle < 0:
                    angle += 360  # Ensure angle is in [0, 360]
                angles.append(angle)
        return angles

    def get_final_heights(self, data_raw, data_whose_y_up=False):
        final_height_list = []
        for traj in data_raw:
            if data_whose_y_up:
                final_height = traj[-1][1]
            else:
                final_height = traj[-1][2]

            # if final_height <= 0.3:
            #     plotter = Plotter()
            #     plotter.plot_samples([positions], title=f'{positions[-1]}', rotate_data_whose_y_up=True)

            final_height_list.append(final_height)
        return final_height_list

    def draw_histogram(self, data, bin_width, x_label, y_label, title, start_x=None, end_x=None):
        # Thiết lập kích thước cửa sổ ngay khi bắt đầu
        fig, ax = plt.subplots(figsize=(20, 12))
        
        # Tính min và max của dữ liệu
        min_value = min(data)
        max_value = max(data)

        # Tạo các bin theo bin_width
        bins = [min_value + i * bin_width for i in range(int((max_value - min_value) / bin_width) + 1)]

        # Tính bin start
        if start_x is not None and start_x < min_value:
            bin_start = np.floor(start_x / bin_width) * bin_width
        else:
            bin_start = np.floor(min_value / bin_width) * bin_width

        # Tính bin end
        if end_x is not None and end_x > max_value:
            bin_end = np.ceil(end_x / bin_width) * bin_width + bin_width
        else:
            bin_end = np.ceil(max_value / bin_width) * bin_width + bin_width

        bins = np.arange(bin_start, bin_end, bin_width)

        # Vẽ histogram
        ax.hist(data, bins=bins, edgecolor='black', alpha=0.7)

        # Ghi giá trị y lên từng bar
        for i in range(len(bins) - 1):
            count = sum(1 for x in data if bins[i] <= x < bins[i + 1])  # Tính tần suất của mỗi bin
            ax.text((bins[i] + bins[i + 1]) / 2, count, str(count), ha='center', va='bottom', fontsize=10, color='red')

        # Tùy chỉnh trục x
        if start_x is not None and end_x is not None:
            ax.set_xlim(start_x, end_x)

        ax.set_xticks(bins)

        # Thiết lập các thuộc tính của biểu đồ
        ax.set_xlabel(x_label, fontsize=14, labelpad=20)
        ax.set_ylabel(y_label, fontsize=14, labelpad=20)
        ax.set_title(title + f' - Total data number: {len(data)}', fontsize=20, pad=30)
        ax.grid(True)

        # Hiển thị biểu đồ
        plt.show()

class RoCatNAEDataRawInvestigator:
    def __init__(self,):
        pass
    
    def plot_start_end_positions(self, trajectories):
        """
        Plot the start and end positions of trajectories.

        Args:
        - trajectories: List of npz-like data, where each item contains 'position' as a key.
                        'position' should be an array of shape (N, 3) representing [x, y, z].

        Returns:
        - None. Displays a 2D plot of start (red) and end (blue) positions.
        """
        start_positions = []
        end_positions = []

        # Extract start and end positions from each trajectory
        for traj in trajectories:
            if 'position' in traj:
                positions = traj['position']
                start_positions.append(positions[0][:2])  # Extract [x, y] for start
                end_positions.append(positions[-1][:2])  # Extract [x, y] for end

        start_positions = np.array(start_positions)
        end_positions = np.array(end_positions)

        # Plot the positions
        plt.figure(figsize=(8, 8))
        plt.scatter(
            start_positions[:, 0], start_positions[:, 1], 
            c='red', label='Start Positions', s=50, edgecolors='black'
        )
        plt.scatter(
            end_positions[:, 0], end_positions[:, 1], 
            c='blue', label='End Positions', s=50, edgecolors='black'
        )

        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("Start and End Positions of Trajectories")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()
    
    def plot_throw_direction_vectors(self, trajectories, data_whose_y_up=False, title_note='', check_starting_point_outlier=False):
        """
        Plot vectors (arrows) from start to end positions of trajectories.

        Args:
        - trajectories: List of npz-like data, where each item contains 'position' as a key.
                        'position' should be an array of shape (N, 3) representing [x, y, z].

        Returns:
        - None. Displays a 2D plot with arrows.
        """
        plt.figure(figsize=(12, 12))
        x_lim_max = -1000
        x_lim_min = 1000
        y_lim_max = -1000
        y_lim_min = 1000

        # Extract and plot vectors
        for idx, traj in enumerate(trajectories):
            if 'position' in traj:
                positions = traj['position']
                if data_whose_y_up:
                    start = [positions[0][0], -positions[0][2], positions[0][1]]  # Start position [x, z]
                    end = [positions[-1][0], -positions[-1][2], positions[0][1]]  # End position [x, z]
                else:
                    start = [positions[0][0], positions[0][1], positions[0][2]]  # Start position [x, z]
                    end = [positions[-1][0], positions[-1][1], positions[0][2]]  # End position [x, z]

                if check_starting_point_outlier:
                    if start[2] >= 1.8:
                        # plot this trajectory
                        plotter = Plotter()
                        plotter.plot_samples([positions], title=f'Outlier trajectory with starting point at [{start[0]:.3f}, {start[1]:.3f}, {start[2]:.3f}]', rotate_data_whose_y_up=True)

                # Draw arrow from start to end
                dx, dy = end[0] - start[0], end[1] - start[1]
                plt.arrow(
                    start[0], start[1], dx, dy, 
                    head_width=0.05, head_length=0.05, fc='green', ec='green', alpha=0.3
                )

                # Mark start position with a red circle
                plt.scatter(
                    start[0], start[1], c='red', s=50, edgecolor='black',
                    label='Start Positions' if idx == 0 else None
                )

                # Update x and y limits
                x_lim_max = max(x_lim_max, start[0], end[0])
                x_lim_min = min(x_lim_min, start[0], end[0])
                y_lim_max = max(y_lim_max, start[1], end[1])
                y_lim_min = min(y_lim_min, start[1], end[1])

        # Set limits for x and y axis
        plt.xlim(x_lim_min - 0.1, x_lim_max + 0.1)
        plt.ylim(y_lim_min - 0.1, y_lim_max + 0.1)
        plt.xlabel("X Position", fontsize=14, labelpad=20)
        plt.ylabel("Y Position", fontsize=14, labelpad=20)
        plt.title(title_note + ' - Throw direction Vectors (Start to End)' + f' - Total data number: {len(trajectories)}', fontsize=20, pad=30)
        plt.grid(True)
        plt.axis('equal')
        plt.legend(loc='upper left')
        plt.show()

    def get_lengths(self, data_raw):
        lengths = [len(traj['position']) for traj in data_raw]
        return lengths
    
    def get_pick_heights(self, data_raw, data_whose_y_up=False):
        pick_height_list = []
        # Extract max heights for each trajectory
        for traj in data_raw:
            if 'position' in traj:
                positions = traj['position']
                if data_whose_y_up:
                    heights = positions[:, 1]  # Use Y coordinate as height
                else:
                    heights = positions[:, 2]  # Use Z coordinate as height
                max_height = np.max(heights)
                pick_height_list.append(max_height)
        return pick_height_list

    def get_final_heights(self, data_raw, data_whose_y_up=False):
        final_height_list = []
        for traj in data_raw:
            if 'position' in traj:
                positions = traj['position']
                if data_whose_y_up:
                    final_height = positions[-1][1]
                else:
                    final_height = positions[-1][2]

                # if final_height <= 0.3:
                #     plotter = Plotter()
                #     plotter.plot_samples([positions], title=f'{positions[-1]}', rotate_data_whose_y_up=True)

                final_height_list.append(final_height)
        return final_height_list
    
    def get_throw_direction_angles(self, data_raw, data_whose_y_up=False):
        angles = []
        # Calculate angles for each trajectory
        for traj in data_raw:
            if 'position' in traj:
                positions = traj['position']
                if data_whose_y_up:
                    start = [positions[0][0], -positions[0][2]]  # Start position [x, y]
                    end = [positions[-1][0], -positions[-1][2]]  # End position [x, y]
                else:
                    start = [positions[0][0], positions[0][1]]
                    end = [positions[-1][0], positions[-1][1]]

                # Compute vector and angle in degrees
                dx, dy = end[0] - start[0], end[1] - start[1]
                angle = np.arctan2(dy, dx) * 180 / np.pi  # Convert radians to degrees
                if angle < 0:
                    angle += 360  # Ensure angle is in [0, 360]
                angles.append(angle)
        return angles

    def draw_histogram(self, data, bin_width, x_label, y_label, title, start_x=None, end_x=None):
        # Thiết lập kích thước cửa sổ ngay khi bắt đầu
        fig, ax = plt.subplots(figsize=(20, 12))
        
        # Tính min và max của dữ liệu
        min_value = min(data)
        max_value = max(data)

        # Tạo các bin theo bin_width
        bins = [min_value + i * bin_width for i in range(int((max_value - min_value) / bin_width) + 1)]

        # Tính bin start
        if start_x is not None and start_x < min_value:
            bin_start = np.floor(start_x / bin_width) * bin_width
        else:
            bin_start = np.floor(min_value / bin_width) * bin_width

        # Tính bin end
        if end_x is not None and end_x > max_value:
            bin_end = np.ceil(end_x / bin_width) * bin_width + bin_width
        else:
            bin_end = np.ceil(max_value / bin_width) * bin_width + bin_width

        bins = np.arange(bin_start, bin_end, bin_width)

        # Vẽ histogram
        ax.hist(data, bins=bins, edgecolor='black', alpha=0.7)

        # Ghi giá trị y lên từng bar
        for i in range(len(bins) - 1):
            count = sum(1 for x in data if bins[i] <= x < bins[i + 1])  # Tính tần suất của mỗi bin
            ax.text((bins[i] + bins[i + 1]) / 2, count, str(count), ha='center', va='bottom', fontsize=10, color='red')

        # Tùy chỉnh trục x
        if start_x is not None and end_x is not None:
            ax.set_xlim(start_x, end_x)

        ax.set_xticks(bins)

        # Thiết lập các thuộc tính của biểu đồ
        ax.set_xlabel(x_label, fontsize=14, labelpad=20)
        ax.set_ylabel(y_label, fontsize=14, labelpad=20)
        ax.set_title(title + f' - Total data number: {len(data)}', fontsize=20, pad=30)

        ax.grid(True)

        # Hiển thị biểu đồ
        plt.show()

def main():
    # investigator = RoCatDataRawInvestigator()
    # rocat_investigator = RoCatRLLabDataRawInvestigator()
    # last_lengths = rocat_investigator.get_final_heights(data_raw, data_whose_y_up=True)
    # # draw histogram
    # bin_width = 0.1
    # x_label = 'Final height'
    # y_label = 'Number of data'
    # title = 'Final height histogram'
    # print('len of last_lengths: ', len(last_lengths))
    # rocat_investigator.draw_histogram(last_lengths, bin_width, x_label, y_label, title)

    data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/data/new_dataset_no_orientation/new_format/3_enrichment/frisbee/frisbee_enriched_267.npz'
    thrown_object = 'check'
    data_raw = RoCatRLLabDataRawReader(data_dir).read_position_data()

    global_util_plotter.plot_trajectory_dataset_plotly(data_raw, title=thrown_object, rotate_data_whose_y_up=False)

if __name__ == '__main__':
    main()