import numpy as np
import pandas as pd
import os

from nae_core.utils.submodules.preprocess_utils.data_raw_reader import RoCatRLLabDataRawReader
from python_utils.plotter import Plotter

class DataFile2Excel:
    def __init__(self, data_file, object_name, data_owner='RLLAB'):
        self.object_name = object_name
        self.data_owner = data_owner
        self.data = RoCatRLLabDataRawReader(data_file).read_raw_data()
        print('Loaded file with ', len(self.data), ' trajectories')

    def save_to_excel(self, need_to_swap_y_z_for_rllab_data=False):
        '''
        save each trajectory to an excel file
        '''
        data_folder = self.create_data_folder()
        for i, traj in enumerate(self.data):
            one_data_dict = {}

            # if self.data_owner == 'NAE':
            #     for key in traj.files:
            #         value = traj[key]
            #         if key == 'frame_num':
            #             one_data_dict['frame_num'] = value
            #         elif key == 'time_step':
            #             one_data_dict['time_step'] = value
            #         elif key == 'position':
            #             one_data_dict['pos_x'] = [v[0] for v in value]
            #             one_data_dict['pos_y'] = [v[1] for v in value]
            #             one_data_dict['pos_z'] = [v[2] for v in value]
            #         elif key == 'quaternion':
            #             one_data_dict['quat_x'] = [v[0] for v in value]
            #             one_data_dict['quat_y'] = [v[1] for v in value]
            #             one_data_dict['quat_z'] = [v[2] for v in value]
            #             one_data_dict['quat_w'] = [v[3] for v in value]

            # dict_keys(['points', 'msg_ids', 'time_stamps', 'low_freq_num'])
            for key, value in traj.items():
                if key == 'msg_ids':
                    one_data_dict['msg_ids'] = value
                elif key == 'time_stamps':
                    one_data_dict['time_stamps'] = value
                elif key == 'points':
                    one_data_dict['pos_x'] = [v[0] for v in value]
                    if need_to_swap_y_z_for_rllab_data:
                        one_data_dict['pos_y'] = [v[2] for v in value]
                        one_data_dict['pos_z'] = [v[1] for v in value]
                    else:
                        one_data_dict['pos_y'] = [v[1] for v in value]
                        one_data_dict['pos_z'] = [v[2] for v in value]
                    one_data_dict['vel_x'] = [v[3] for v in value]
                    if need_to_swap_y_z_for_rllab_data:
                        one_data_dict['vel_y'] = [v[5] for v in value]
                        one_data_dict['vel_z'] = [v[4] for v in value]
                    else:
                        one_data_dict['vel_y'] = [v[4] for v in value]
                        one_data_dict['vel_z'] = [v[5] for v in value]

            one_data_df = pd.DataFrame(one_data_dict)

            if self.data_owner == 'RLLAB':
                desired_order = ['msg_ids', 'time_stamps'] + [col for col in one_data_df.columns if col not in ['msg_ids', 'time_stamps']]
                one_data_df = one_data_df[desired_order]
        
            output_file = os.path.join(data_folder, f'{self.object_name}_{i}.xlsx')
            one_data_df.to_excel(output_file, index=False)
            print('Saved to ', output_file)

    def create_data_folder(self,):
        # get directory of this script
        data_folder = os.path.dirname(os.path.realpath(__file__))
        # cd ..
        data_folder = os.path.dirname(data_folder)
        # cd to data folder
        data_folder = os.path.join(data_folder, 'data', 'excel_data_files', self.data_owner+'_dataset_excel', self.object_name)
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        return data_folder

# main
def main():
    ## ================= RLLAB dataset =================

    data_owner = 'RLLAB'


    ## ======== NAE new data format
    # data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/mocap_ws/src/mocap_data_collection/data/frisbee_ring/min_len_65/15-01-2025_10-42-46-traj_num-102.npz'
    # object_name = 'ring_frisbee'

    data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/mocap_ws/src/mocap_data_collection/data/empty_bottle/min_len_50/16-01-2025_16-50-11-traj_num-100.npz'
    object_name = 'empty_bottle'

    data_to_excel = DataFile2Excel(data_dir,object_name=object_name)
    '''
    need_to_swap_y_z_for_rllab_data: when colleting data, we swapped y and z axis for rllab data, however, after that, we decided to keep the original data format with y axis as up axis
    so we need to swap y and z axis back to the original format by setting need_to_swap_y_z_for_rllab_data=True
    '''
    data_to_excel.save_to_excel(need_to_swap_y_z_for_rllab_data=False)

if __name__ == '__main__':
    main()
