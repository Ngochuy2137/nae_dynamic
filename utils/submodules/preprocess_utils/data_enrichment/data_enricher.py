import numpy as np
import torch 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os 
import random

from nae_core.utils.submodules.preprocess_utils.data_enrichment.data_points_rotator import DataPointsRotator
from nae_core.utils.submodules.preprocess_utils.data_raw_reader import RoCatRLLabDataRawReader, RoCatNAEDataRawReader
from nae_core.utils.submodules.preprocess_utils.data_raw_correction_checker import RoCatRLDataRawCorrectionChecker
from nae_core.utils.submodules.preprocess_utils.data_raw_analyst import RoCatRLLabDataRawInvestigator, RoCatNAEDataRawInvestigator
from nae_core.utils.submodules.preprocess_utils.data_splitter import RoCatDataSplitter
from python_utils.plotter import Plotter

class DataEnricher:
    def __init__(self, object_name):
        self.object_name = object_name
        self.data_points_rotator = DataPointsRotator()

    def enrich_dataset(self, data, angle_deg_rot=0, axis_rot=(np.array([0, 0, 0]), np.array([0, 1, 0])), vector_trans=np.array([0, 0, 0])):
        '''
        Enrich dataset by:
            Rotate the dataset (many trajectories) around an axis_rot by a specific angle.
            Translate the dataset by a specific vector.
        Args:
            data (list): List of dictionaries containing the data of the trajectories.
            angle_deg_rot (float): The rotation angle in degrees.
            axis_rot (tuple): Tuple containing the origin and direction of the rotation axis.
        '''

        angle_rad_rot = np.deg2rad(angle_deg_rot)
        axis_origin, axis_direction = axis_rot

        vector_trans = np.concatenate([np.array(vector_trans), np.zeros(6)])
        # add 0, 0, 0 to the vector_trans if it has less than 3 elements
        # print(data[0].keys())       # 'points', 'msg_ids', 'time_stamps', 'low_freq_num'
        data_enriched = []
        for d in data:
            d_points = d['points']
            d_msg_ids = d['msg_ids']
            d_time_stamps = d['time_stamps']
            if 'low_freq_num' in d.keys():
                d_low_freq_num = d['low_freq_num']

            # Step 1: Rotate the dataset
            traj_rotated = self.data_points_rotator.rotate_points_around_axis(d_points, (axis_origin, axis_direction), angle_rad_rot)
            # Step 2: Translate the dataset
            traj_enriched = traj_rotated + vector_trans

            if 'low_freq_num' in d.keys():
                data_point_rotated = {
                    'points': traj_enriched,
                    'msg_ids': d_msg_ids,
                    'time_stamps': d_time_stamps,
                    'low_freq_num': d_low_freq_num
                }
            else:
                data_point_rotated = {
                    'points': traj_enriched,
                    'msg_ids': d_msg_ids,
                    'time_stamps': d_time_stamps,
                }
            data_enriched.append(data_point_rotated)

        return data_enriched
    
    def save_dataset(self, data, angle_deg_rot, axis_rot, vector_trans, original_data_dir):
        # get current path
        current_path = os.path.dirname(os.path.realpath(__file__))
        # get parent path
        parent_path = os.path.abspath(os.path.join(current_path, os.pardir))

        # prepare the new data folder
        new_data_folder = os.path.join(parent_path, 'data', self.object_name, 'data_enrichment')
        if not os.path.exists(new_data_folder):
            os.makedirs(new_data_folder)
        data_path = os.path.join(new_data_folder, self.object_name + f'_enriched_{len(data)}.npz')

        # prepare data to save
        new_data_dict = {'trajectories': data,
                        'object_name': self.object_name}
        
        # save the data
        np.savez(data_path, **new_data_dict)

        print(f'Enriched data is saved to {data_path}')
        self.save_info(angle_deg_rot, axis_rot, vector_trans, new_data_folder, original_data_dir)

    def save_info(self, angle_deg_rot, axis_rot, vector_trans, save_dir, original_data_dir):
        '''
        Save enrichment information to a txt file
        '''
        axis_origin = axis_rot[0]
        axis_direction = axis_rot[1]

        x_trans = vector_trans[0]
        y_trans = vector_trans[1]
        z_trans = vector_trans[2]
        info_file = os.path.join(save_dir, 'info_enrichment.txt')
        with open(info_file, 'w') as f:
            f.write(f'Object name: {self.object_name}\n')
            f.write(f'Rotation:\n')
            f.write(f'      Rotation angle: {angle_deg_rot}\n')
            f.write(f'      Axis origin: {axis_origin}\n')
            f.write(f'      Axis direction: {axis_direction}\n')
            f.write(f'Translation vector: {x_trans}, {y_trans}, {z_trans}\n')
            f.write(f'Original data dir: {original_data_dir}\n')
        print(f'Enrichment information is saved to {info_file}')
            

if __name__ == "__main__":
    # data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/data/new_dataset_no_orientation/new_format/2_new_format/big_plane/big_plane_new_data_format_num_257.npz'
    # object_name = 'big_plane'

    # data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/data/new_dataset_no_orientation/new_format/2_new_format/frisbee/frisbee_new_data_format_num_267.npz'
    # object_name = 'frisbee'

    data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/data/new_dataset_no_orientation/new_format/2_new_format/boomerang/boomerang_new_data_format_num_251.npz'
    object_name = 'boomerang'

    data = RoCatRLLabDataRawReader(data_dir).read_raw_data()
    data_enricher = DataEnricher(object_name)
    data_analyst = RoCatRLLabDataRawInvestigator()
    # ----------------------------------------------------------
    # params for rotate and translate NAE data 90 degree to test
    angle_deg_rot = 180
    # axis_origin = np.array([1.5, 0, 1.0])
    axis_origin = np.array([2, 0, 0.25])
    axis_direction = np.array([0, 1, 0])
    x_trans = 0.0
    y_trans = 0.0
    z_trans = 0.0

    # shuffle data
    for _ in range(5):
        random.shuffle(data)
    # choose 1/2 data to enrich
    data_first_half = data[:int(len(data)/2)]
    data_second_half = data[int(len(data)/2):]
    data_second_half_enriched = data_enricher.enrich_dataset(data_second_half, 
                                                        angle_deg_rot=angle_deg_rot, 
                                                        axis_rot=(axis_origin, axis_direction), 
                                                        vector_trans=[x_trans, y_trans, z_trans]
                                                        )
    data_analyst.plot_throw_direction_vectors(data_first_half, data_whose_y_up=True, title_note=f'{object_name}\ndata_first_half')
    data_analyst.plot_throw_direction_vectors(data_second_half, data_whose_y_up=True, title_note=f'{object_name}\ndata_second_half')
    data_analyst.plot_throw_direction_vectors(data_second_half_enriched, data_whose_y_up=True, title_note=f'{object_name}\ndata_second_half_enriched')
    
    # merge the enriched data with the rest of the data
    new_dataset = list(data_first_half) + data_second_half_enriched
    data_analyst.plot_throw_direction_vectors(new_dataset, data_whose_y_up=True, title_note=f'{object_name}\ndata enriched')

    # save
    input('Press ENTER to save the enriched data...')
    data_enricher.save_dataset(new_dataset, 
                               angle_deg_rot = angle_deg_rot, 
                               axis_rot = (axis_origin, axis_direction),  
                               vector_trans=[x_trans, y_trans, z_trans], 
                               original_data_dir=data_dir)

    data_collection_checker = RoCatRLDataRawCorrectionChecker()
    data_collection_checker.check_data_correction(new_dataset, data_whose_y_up=True)

    input('Do you want to split data?')
    data_splitter = RoCatDataSplitter(new_dataset, train_ratio=0.8, val_ratio=0.1, num_first_points_to_cut=0, object_name=object_name)
    data_splitter.split(shuffle_data=True) 

