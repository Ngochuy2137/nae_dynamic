import numpy as np
from nae_core.utils.submodules.preprocess_utils.data_raw_reader import RoCatRLLabDataRawReader, RoCatNAE_coreDataRawReader
from nae_core.utils.submodules.plotter import RoCatDataPlotter
import os

class RoCatRLLabDataMerger:
    def __init__(self, data_folder, object_name, file_format='npz'):
        self.data_files = []
        self.trajectories = []
        self.object_name = object_name
        self.data_files = self.look_for_all_data_files(data_folder, file_format)
        self.merge()
        
    def merge(self):
        self.trajectories = []
        print(f'Merging data from {len(self.data_files)} files')
        for file in self.data_files:
            data_reader = RoCatRLLabDataRawReader(file)
            self.trajectories.extend(data_reader.read())
        print('-> Done merging')
        print(f'Total trajectories: {len(self.trajectories)}')
    
    def look_for_all_data_files(self, data_folder, file_format='npz'):
        matched_files = []
        count = 0
        for root, _, files in os.walk(data_folder):
            for filename in files:
                if filename.endswith(file_format):
                    matched_files.append(os.path.join(root, filename))
                    count += 1
                    print(f'    Found file: {filename}')
        print(f'-> Found {count} files')
        return matched_files
    
    def get_data(self):
        return self.trajectories
    
    def analyze(self, threshold=65):
        # check trajectory with min, max length
        len_list = [len(traj['points']) for traj in self.trajectories]
        min_len = min(len_list)
        max_len = max(len_list)
        print(f'Min length: {min_len}, Max length: {max_len}')

        # check number of trajectories with length > threshold
        len_list_gt_threshold = [len(traj['points']) for traj in self.trajectories if len(traj['points']) > threshold]
        print(f'Number of trajectories with length > {threshold}: {len(len_list_gt_threshold)}')

    def filter(self, threshold=65):
        self.trajectories = [traj for traj in self.trajectories if len(traj['points']) > threshold]
        print(f'Filtered out trajectories with length <= {threshold}')
        print(f'Number of remaining trajectories: {len(self.trajectories)}')

    # save to a new npz file
    def save(self, ):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
        save_folder = os.path.join(parent_dir, 'data', 'merged', self.object_name)
        os.makedirs(save_folder, exist_ok=True)
        save_file = os.path.join(save_folder, f'{self.object_name}_merged_{len(self.trajectories)}.npz')
        np.savez(save_file, trajectories=self.trajectories)
        print(f'Saved {len(self.trajectories)} merged data to {save_file}')


class RoCatNAEDataMerger:
    def __init__(self, data_folder, object_name, file_format='npy'):
        self.data_files = []
        self.trajectories = []
        self.object_name = object_name
        self.data_files = self.look_for_all_data_files(data_folder, file_format)
        self.merge()
        
    def merge(self):
        self.trajectories = []
        print(f'Merging data from {len(self.data_files)} files')
        for file in self.data_files:
            data_reader = RoCatNAEDataRawReader(file)
            self.trajectories.extend(data_reader.read())
        print('-> Done merging')
        print(f'Total trajectories: {len(self.trajectories)}')
    
    def look_for_all_data_files(self, data_folder, file_format='npy'):
        matched_files = []
        count = 0
        for root, _, files in os.walk(data_folder):
            for filename in files:
                if filename.endswith(file_format):
                    matched_files.append(os.path.join(root, filename))
                    count += 1
                    print(f'    Found file: {filename}')
        print(f'-> Found {count} files')
        return matched_files
    
    def get_data(self):
        return self.trajectories
    
# main function
def main():
    data_folder = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_prediction_ws/src/nae/data/frisbee-pbl'
    object_name = 'frisbee-pbl'
    data_merger = RoCatRLLabDataMerger(data_folder, object_name)
    data_merger.analyze()
    data_merger.filter()
    trajectories = data_merger.get_data()

    print('Trajectories:', len(trajectories))
    # data_merger.save()
    # plot
    plotter = RoCatDataPlotter()
    for traj in trajectories:
        plotter.plot_samples([traj['points']], title='Trajectory')
        

# call main function
if __name__ == '__main__':
    main()
