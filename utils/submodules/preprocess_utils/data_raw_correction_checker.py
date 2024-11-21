#!/usr/bin/env python3

'''
Program description:
    - Read npz file and plot the trajectory of the object.
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import random
from python_utils.printer import Printer
DEBUG = False
np.set_printoptions(suppress=True)
from nae.utils.submodules.preprocess_utils.data_raw_reader import RoCatRLLabDataRawReader

class RoCatRLDataRawCorrectionChecker:
    def __init__(self,):
        self.util_printer = Printer()
    '''
    check if the interpolation is correct
    Get 10 random trajectories from the trajectory and check if the velocity is correct
    '''
    def check_data_correction(self, data, new_data_format=False):
        '''
        check swap_yz: If the data is swapped y and z, g should be the 8th element in one point
        get 10 random trajectories
            The old data format is:
            - Swap y, z
            - Swap vy, vz
            - Data point: x, y, z, vx, vy, vz, 0, 0, 9.81

            The new data format is:
            - Keep y, z, no swap anymore => need to:
                + Swap y, z
                + Swap vy, vz
            - Data point: x, y, z, vx, vy, vz, 0, -9.81, 0
        '''

        sample_size = 10
        if len(data) < 10:
            print('Checking less data ({len(data)}) than expected')
            sample_size = len(data)
        
        indices = random.sample(range(len(data)), sample_size)

        count = 0
        for i in indices:
            # print a random point to check
            # print in blue background
            print('\n')
            self.util_printer.print_blue(f'{count} ----- checking trajectory {i} with {len(data[i]["points"])} points -----', background=True)
            count += 1
            point_random_idx = random.randint(0, len(data[i]) - 1)
            print(f'Trajectory {i} -> point {point_random_idx}:')
            print('     - Point:        ', data[i]['points'][point_random_idx])
            # print('     - Orientation:  ', data[i]['orientations'][point_random_idx])
            print('     - Timestamp:    ', data[i]['time_stamps'][point_random_idx])
            result_feature_check, msg = self.check_feature_correction(data[i], new_data_format)
            if not result_feature_check:
                # print in red
                self.util_printer.print_red(f'[FEATURE CHECK] Trajectory {i} has incorrect data')
                print('     ', msg)
            # print in green
            self.util_printer.print_green(f'[FEATURE CHECK]             Trajectory {i} has correct data')
        
            if not self.check_velocity_correction(data[i]):
                print(f'\033[91m[VEL INTERPOLATION CHECK] Trajectory {i} has incorrect data')
            # print in green
            self.util_printer.print_green(f'[VEL INTERPOLATION CHECK]   Trajectory {i} has correct data')
        
            
    def check_feature_correction(self, one_trajectory, new_data_format=False):
        '''
        The old data format is:
            - Swap y, z
            - Swap vy, vz
            - Data point: x, y, z, vx, vy, vz, 0, 0, 9.81

        The new data format is:
            - Keep y, z, no swap anymore => need to:
                + Swap y, z
                + Swap vy, vz
            - Data point: x, y, z, vx, vy, vz, 0, -9.81, 0
        '''
        for point in one_trajectory['points']:
            if new_data_format:
                is_feature_7_right = abs(point[7] + 9.81) < 1e-5
                is_feature_8_right = abs(point[8]) < 1e-5
            else:
                is_feature_7_right = abs(point[7]) < 1e-5
                is_feature_8_right = abs(point[8] - 9.81) < 1e-5
            if not is_feature_7_right or not is_feature_8_right:
                msg = point
                return False, msg
        return True, ''
    def check_velocity_correction(self, one_trajectory):
        """
        Kiểm tra xem nội suy vận tốc có đúng không.
        The proper velocities vx, vy, vz should follow: 
        - Forward difference formula for the first point
        - Backward difference formula for the last point
        - Central difference formula for the rest of the points
        """
        traj_len = len(one_trajectory['points'])
        for i in range(traj_len):
            point = one_trajectory['points'][i]
            timestamp = one_trajectory['time_stamps'][i]
            if i == 0:
                # Forward difference
                next_point = one_trajectory['points'][i + 1]
                next_timestamp = one_trajectory['time_stamps'][i + 1]
                dt = next_timestamp - timestamp

                vx_expected = (next_point[0] - point[0]) / dt
                vy_expected = (next_point[1] - point[1]) / dt
                vz_expected = (next_point[2] - point[2]) / dt
            elif i == traj_len - 1:
                # Backward difference
                prev_point = one_trajectory['points'][i - 1]
                prev_timestamp = one_trajectory['time_stamps'][i - 1]
                dt = timestamp - prev_timestamp

                vx_expected = (point[0] - prev_point[0]) / dt
                vy_expected = (point[1] - prev_point[1]) / dt
                vz_expected = (point[2] - prev_point[2]) / dt
            else:
                # Central difference
                prev_point = one_trajectory['points'][i - 1]
                next_point = one_trajectory['points'][i + 1]
                prev_timestamp = one_trajectory['time_stamps'][i - 1]
                next_timestamp = one_trajectory['time_stamps'][i + 1]
                dt = next_timestamp - prev_timestamp

                vx_expected = (next_point[0] - prev_point[0]) / dt
                vy_expected = (next_point[1] - prev_point[1]) / dt
                vz_expected = (next_point[2] - prev_point[2]) / dt

            # Lấy vận tốc thực tế từ dữ liệu
            vx, vy, vz = point[3], point[4], point[5]

            # Kiểm tra nếu vận tốc khớp với sai số cho phép
            tolerance = 1e-5
            result = abs(vx - vx_expected) < tolerance and \
                    abs(vy - vy_expected) < tolerance and \
                    abs(vz - vz_expected) < tolerance
            if not result:
                print(f'\033[91m[VEL INTERPOLATION CHECK] Point {i} has incorrect velocity')
                print(f'Expected: vx = {vx_expected}, vy = {vy_expected}, vz = {vz_expected}')
                if i == 0:
                    print('Forward difference')
                    print('     ', one_trajectory['points'][i][:6])
                    print('         ', one_trajectory['time_stamps'][i])
                    print('     ', one_trajectory['points'][i + 1][:6])
                    print('         ', one_trajectory['time_stamps'][i + 1])
                elif i == traj_len - 1:
                    print('Backward difference')
                    print('     ', one_trajectory['points'][i - 1][:6])
                    print('         ', one_trajectory['time_stamps'][i - 1])
                    print('     ', one_trajectory['points'][i][:6])
                    print('         ', one_trajectory['time_stamps'][i])
                else:
                    print('Central difference')
                    print('     ', one_trajectory['points'][i - 1][:6])
                    print('         ', one_trajectory['time_stamps'][i-1])
                    print('     ', one_trajectory['points'][i][:6])
                    print('         ', one_trajectory['time_stamps'][i])
                    print('     ', one_trajectory['points'][i + 1][:6])
                    print('         ', one_trajectory['time_stamps'][i + 1])
                input()
            #print in green
            if i == 0:
                cal_way = 'Forward difference'
            elif i == traj_len - 1:
                cal_way = 'Backward difference'
            else:
                cal_way = 'Central difference'
            self.util_printer.print_green(f'[VEL INTERPOLATION CHECK] pass - {cal_way}', enable=DEBUG)
        return result
                
# main
if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    # Create a path to the directory ../trajectories
    # file_path = os.path.join(parent_dir, 'data/frisbee-pbl', 'frisbee-pbl_merged_275.npz')
    file_path='/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_prediction_ws/src/nae/data/frisbee-pbl/frisbee-pbl_merged_275.npz'
    data_reader = RoCatRLLabDataRawReader(file_path)
    data_collection_checker = RoCatRLDataRawCorrectionChecker()
    data_collection_checker.check_data_correction(data_reader.read(), new_data_format=True)

    one_trajectory = data_reader.read()[1]
    print('check 111: ', one_trajectory['points'].shape)
    print('check 222: ', one_trajectory['msg_ids'].shape)
    print('check 333: ', one_trajectory['time_stamps'].shape)
    print('check 444: low_freq_num value ', one_trajectory['low_freq_num'])

    input('1')
    data_reader.check_data_correction()