import numpy as np
import torch 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

from python_utils.printer import Printer

class DataPointsRotator:
    def __init__(self):
        self.util_printer = Printer()

    def rotate_points_around_axis(self, data_points, axis_rot, angle_rad):
        """
        Rotate points and velocity vectors around a given axis_rot with a specific origin and direction.

        Args:
            points (np.ndarray): Array of shape (N, 3) containing the points.
            velocities (np.ndarray): Array of shape (N, 3) containing the velocity vectors.
            axis_origin (np.ndarray): Array of shape (3,) representing the origin of the rotation axis_rot.
            axis_direction (np.ndarray): Array of shape (3,) representing the direction of the rotation axis_rot (must be normalized).
            angle_rad (float): The rotation angle_rad in radians.

        Returns:
            tuple: Rotated points and velocity vectors as numpy arrays of shape (N, 3).
        """
        axis_origin, axis_direction = axis_rot

        points = data_points[:, :3]
        vel_rotate = acc_rotate = False  # Initialize flags
        velocities = accelerations = None  # Initialize optional arrays

        if data_points.shape[1] == 3:
            vel_rotate = False
            acc_rotate = False
        elif data_points.shape[1] == 6:
            self.util_printer.print_blue('Data WITH velocities')
            velocities = data_points[:, 3:6]
            vel_rotate = True
            acc_rotate = False
        elif data_points.shape[1] == 9:
            self.util_printer.print_blue('Data WITH accelerations')
            velocities = data_points[:, 3:6]
            accelerations = data_points[:, 6:9]
            vel_rotate = True
            acc_rotate = True


        # Normalize the axis direction
        # axis_direction = axis_direction / torch.linalg.norm(axis_direction)
        axis_direction = axis_direction / np.linalg.norm(axis_direction)

        # Step 1: Translate points so that axis_origin becomes the origin
        translated_points = points - axis_origin

        # Step 2: Create the rotation object
        rotation = R.from_rotvec(angle_rad * axis_direction)  # Rotation based on axis and angle_rad

        # Step 3: Rotate the points and velocities
        rotated_translated_points = rotation.apply(translated_points)
        if vel_rotate:
            rotated_velocities = rotation.apply(velocities)  # Velocities are rotated directly
        if acc_rotate:
            rotated_accelerations = rotation.apply(accelerations)

        # Step 4: Translate points back to the original position
        rotated_points = rotated_translated_points + axis_origin

        # Step 5: Merge points and velocities
        if vel_rotate:
            rotated_points = np.concatenate([rotated_points, rotated_velocities], axis=1)
        if acc_rotate:
            rotated_points = np.concatenate([rotated_points, rotated_accelerations], axis=1)
        return rotated_points

    def plot_points_rotation(self, original_points, rotated_points, axis_rot, labels=None):
        """
        Plot points in 3D space before and after rotation, with automatic labels, rotation axis_rot,
        dashed lines to the axis_rot, and angle markers. The rotation axis_rot is represented by an arrow.

        Args:
            original_points (torch.Tensor): Tensor of shape (N, 3) containing original points.
            rotated_points (torch.Tensor): Tensor of shape (N, 3) containing rotated points.
            axis_rot (tuple): A tuple of (axis_origin, axis_direction) defining the rotation axis_rot.
            labels (list of str): (Optional) Custom list of labels for points. If None, auto-generate labels.
        """
        # Convert axis_origin and axis_direction to numpy (from GPU to CPU)
        axis_origin = axis_rot[0]
        axis_direction = axis_rot[1]

        # Convert points to numpy for plotting
        original_points = original_points[:, :3]
        rotated_points = rotated_points[:, :3]

        # Normalize the rotation axis direction
        axis_direction = axis_direction / np.linalg.norm(axis_direction)

        # Create a 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the rotation axis as an arrow
        arrow_length = 2.0  # Length of the arrow
        ax.quiver(
            axis_origin[0], axis_origin[1], axis_origin[2],  # Starting point of the arrow
            axis_direction[0], axis_direction[1], axis_direction[2],  # Direction of the arrow
            length=arrow_length, color='green', label='Rotation Axis', arrow_length_ratio=0.1
        )

        # Plot original and rotated points, and draw dashed lines to the axis
        for i, (orig, rot) in enumerate(zip(original_points, rotated_points)):
            # Compute projection of points onto the rotation axis
            projection_orig = axis_origin + np.dot(orig - axis_origin, axis_direction) * axis_direction
            projection_rot = axis_origin + np.dot(rot - axis_origin, axis_direction) * axis_direction

            # Plot the points
            ax.scatter(orig[0], orig[1], orig[2], color='blue', label='Original Point' if i == 0 else "")
            ax.scatter(rot[0], rot[1], rot[2], color='red', label='Rotated Point' if i == 0 else "")

            # Automatic labels
            label = f"point{i+1}" if labels is None else labels[i]
            ax.text(orig[0], orig[1], orig[2], label, color='blue', fontsize=10)
            ax.text(rot[0], rot[1], rot[2], f"{label}'", color='red', fontsize=10)

            # Draw dashed lines to the axis
            ax.plot([orig[0], projection_orig[0]], [orig[1], projection_orig[1]], [orig[2], projection_orig[2]],
                    linestyle='--', color='gray')
            ax.plot([rot[0], projection_rot[0]], [rot[1], projection_rot[1]], [rot[2], projection_rot[2]],
                    linestyle='--', color='gray')

            # Draw angle markers
            angle_rad = np.arccos(np.dot(orig - projection_orig, rot - projection_rot) /
                                (np.linalg.norm(orig - projection_orig) * np.linalg.norm(rot - projection_rot)))
            angle_deg = np.degrees(angle_rad)

            # Midpoint for placing the angle marker
            mid_point = (orig + rot) / 2
            ax.text(mid_point[0], mid_point[1], mid_point[2], f"{angle_deg:.1f}°", color='purple', fontsize=10)

        # Set labels and title
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title('Points in 3D Space with Rotation Axis and Angles')
        ax.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    # Define points (N, 3) - moving to GPU
    points = np.array([
        [1.0, 1.0, 1.0, 1, 1, 1, 0, -9.8, 0],
        [2.0, 2.0, 0.0, 1, 1, 1, 0, -9.8, 0],
        [3.0, 0.0, 0.0, 1, 1, 1, 0, -9.8, 0]
    ])  # Move points to GPU

    # Define axis_direction, angle, and axis_direction position
    axis_origin = np.array([1.0, 0.0, 3.0])  # Axis passes through this point
    axis_direction = np.array([0.0, 1.0, 0.0])  # Rotation around z-axis_direction
    angle = np.array(3.14159 / 2)  # Rotate by 45 degrees


    data_enricher = DataPointsRotator()
    # Rotate points
    rotated_points = data_enricher.rotate_points_around_axis(points, (axis_origin, axis_direction), angle)

    # Print results
    print(rotated_points)  # Move results back to CPU for display

    # Plot points in 3D space
    data_enricher.plot_points_rotation(points, rotated_points, (axis_origin, axis_direction))