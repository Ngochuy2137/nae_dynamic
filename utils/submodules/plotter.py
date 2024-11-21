import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class RoCatDataPlotter:
    def __init__(self):
        self.colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'pink', 'brown', 'gray']
        self.simbol = ['o', '+', 'x', 's', 'p', 'h', 'd', 'v', '^', '<', '>', '1', '2', '3', '4', '8']
        
    def plot_samples(self, samples, title=''):
        print('Plotting samples...')
        figure = plt.figure(num=1)
        ax = figure.add_subplot(111, projection='3d')
        limit = min(len(self.colors), len(samples))
        x_min = 1000
        x_max = -1000
        y_min = 1000
        y_max = -1000
        z_min = 1000
        z_max = -1000
        for i in range (limit-1):
            sample = np.array(samples[i])
            limit = len(self.colors)
            if i+1 >= limit:
                return
            ax.plot(sample[:, 0], sample[:, 1], sample[:, 2], 
                        'o', color=self.colors[i], alpha=0.5, label='Test '+str(i+1) + 'Sample Trajectory')
            x_min = min(x_min, sample[:, 0].min())
            x_max = max(x_max, sample[:, 0].max())
            y_min = min(y_min, sample[:, 1].min())
            y_max = max(y_max, sample[:, 1].max())
            z_min = min(z_min, sample[:, 2].min())
            z_max = max(z_max, sample[:, 2].max())
            
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.legend()
        plt.title('3D samples ' + title, fontsize=25)
        plt.show()
    
    def update_plot(self, one_trajectory_prediction, one_trajectory_label, color_idx, ax, swap_y_z=False):
        # Check if the color index is within the range
        if color_idx+1 >= len(self.colors):
            return

        # Determine axis order based on swap_y_z flag
        if swap_y_z:
            x_idx, y_idx, z_idx = 0, 2, 1  # Swap y and z
        else:
            x_idx, y_idx, z_idx = 0, 1, 2  # Default order

        # Plot actual and predicted trajectories
        ax.plot(one_trajectory_label[:, x_idx], one_trajectory_label[:, y_idx], one_trajectory_label[:, z_idx],
                    'o', color=self.colors[color_idx], alpha=0.5, label=f'Test {color_idx + 1} Actual Trajectory', markersize=10)
        ax.plot(one_trajectory_prediction[:, x_idx], one_trajectory_prediction[:, y_idx], one_trajectory_prediction[:, z_idx],
                    '+', color=self.colors[color_idx], label=f'Test {color_idx + 1} Predicted Trajectory', markersize=10)

        # Calculate limits
        all_data = np.concatenate((one_trajectory_label, one_trajectory_prediction), axis=0)
        ax.set_xlim([all_data[:, x_idx].min(), all_data[:, x_idx].max()])
        ax.set_ylim([all_data[:, y_idx].min(), all_data[:, y_idx].max()])
        ax.set_zlim([all_data[:, z_idx].min(), all_data[:, z_idx].max()])

        # Set labels based on axis order
        ax.set_xlabel('X')
        ax.set_ylabel('Z' if swap_y_z else 'Y')
        ax.set_zlabel('Y' if swap_y_z else 'Z')

        # Calculate and display the distance for the last points
        actual_point = one_trajectory_label[-1, [x_idx, y_idx, z_idx]]
        predicted_point = one_trajectory_prediction[-1, [x_idx, y_idx, z_idx]]
        distance = np.linalg.norm(actual_point - predicted_point)

        # Plot the distance between the last points with a line and annotate the value
        ax.plot([actual_point[0], predicted_point[0]],
                    [actual_point[1], predicted_point[1]],
                    [actual_point[2], predicted_point[2]],
                    linestyle='--', color='red', label=f'Error Distance: {distance:.2f}')
        
        # Annotate the distance value with a larger fontsize
        mid_point = (actual_point + predicted_point) / 2
        ax.text(mid_point[0], mid_point[1], mid_point[2], f'{distance:.2f}', color='red', fontsize=20)  # Change fontsize here

    def plot_predictions(self, predictions, test_labels, lim_plot_num=None, swap_y_z=False, title=''):
        if lim_plot_num:
            predictions = predictions[:lim_plot_num]
            test_labels = test_labels[:lim_plot_num]
        lim_plot_num = min(len(self.colors), len(predictions))
        figure = plt.figure()
        ax = figure.add_subplot(111, projection='3d')
        for i in range (lim_plot_num):
            self.update_plot(predictions[i], test_labels[i], i, ax, swap_y_z=False)

        plt.legend()
        plt.title('3D Predictions ' + title, fontsize=18)
        plt.show()