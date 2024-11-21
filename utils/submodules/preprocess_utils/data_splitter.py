import numpy as np
import os
from data_raw_reader import RoCatRLLabDataRawReader

class RoCatDataSplitter:
    def __init__(self, data_raw, train_ratio=0.8, val_ratio=0.1, num_first_points_to_cut=0, object_name=None):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.object_name = object_name
        self.data_raw = data_raw
        self.data_raw = np.array([traj['points'][num_first_points_to_cut:] for traj in self.data_raw], dtype=object)
        print('Data len: ', len(self.data_raw))

        print(f'Sample data point: {self.data_raw[0][0]}')

    def split(self, shuffle_data):
        if shuffle_data:
            for i in range(10):
                np.random.shuffle(self.data_raw)

        num_train = int(len(self.data_raw) * self.train_ratio)
        num_val = int(len(self.data_raw) * self.val_ratio)

        self.data_train = self.data_raw[:num_train]
        self.data_val = self.data_raw[num_train:num_train+num_val]
        self.data_test = self.data_raw[num_train+num_val:]

        print('Train data shape: ', len(self.data_train))
        print('Val data shape: ', len(self.data_val))
        print('Test data shape: ', len(self.data_test))
        print('\n')        
        # get current directory, parent directory, and save folder
        current_dir = os.path.dirname(os.path.realpath(__file__))
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
        self.output_data_dir = os.path.join(parent_dir, 'data', 'split', self.object_name)
        if not os.path.exists(self.output_data_dir):
            os.makedirs(self.output_data_dir)

        # save to .npy files
        np.save(os.path.join(self.output_data_dir, f'data_train_{len(self.data_train)}.npy'), self.data_train, allow_pickle=True)
        np.save(os.path.join(self.output_data_dir, f'data_val_{len(self.data_val)}.npy'), self.data_val, allow_pickle=True)
        np.save(os.path.join(self.output_data_dir, f'data_test_{len(self.data_test)}.npy'), self.data_test, allow_pickle=True)


        # np.savez(os.path.join(self.output_data_dir, f'data_train_{len(self.data_train)}.npz'), data=self.data_train)
        # np.savez(os.path.join(self.output_data_dir, f'data_val_{len(self.data_val)}.npz'), data=self.data_val)
        # np.savez(os.path.join(self.output_data_dir, f'data_test_{len(self.data_test)}.npz'), data=self.data_test)
        print(f'Saved train, val, test data to {self.output_data_dir}')

        return self.data_train, self.data_val, self.data_test

def main():
    object_name = 'frisbee-pbl'
    data_path = "/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_prediction_ws/src/nae/utils/submodules/data/frisbee-pbl/frisbee-pbl_new_data_format_num_275.npz"

    data_raw = RoCatRLLabDataRawReader(data_path).read()

    # split data
    data_splitter = RoCatDataSplitter(data_raw, train_ratio=0.8, val_ratio=0.1, num_first_points_to_cut=0, object_name=object_name)
    data_splitter.split(shuffle_data=True)  


# call main function
if __name__ == '__main__':
    main()