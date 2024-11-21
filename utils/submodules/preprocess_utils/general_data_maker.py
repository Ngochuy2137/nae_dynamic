import numpy as np
import os
from nae.utils.submodules.training_utils.data_loader import DataLoader
from python_utils.printer import Printer

class GeneralDataMaker:
    '''
    Mix training data from different objects
    Mix validation data from different objects
    Mix testing data from different objects
    '''

    def __init__(self, data_info):
        self.data_info = data_info
        self.util_printer = Printer()

    def mix_data(self, object_name_list):
        data_train_mix = []
        data_val_mix = []
        data_test_mix = []

        count = 0
        print(f'Loading data from {len(self.data_info)} folders')
        for object_name, data_folder in self.data_info:
            print(f'\n------------------- [{count}] -------------------')
            count += 1
            self.util_printer.print_blue(f'Object name: {object_name}', background=True)
            print('     path: ', data_folder)

            if not os.path.exists(data_folder):
                raise FileNotFoundError(f"Folder {data_folder} does not exist")
            loader = DataLoader()
            data_train, data_val, data_test = loader.load_dataset(data_folder)
            print(f'Found:')
            print(f'  Training data     : {len(data_train)}')
            print(f'  Validation data   : {len(data_val)}')
            print(f'  Testing data      : {len(data_test)}')

            # append data to the mix
            data_train_mix.append(data_train)
            data_val_mix.append(data_val)
            data_test_mix.append(data_test)

        data_train_mix = np.concatenate(data_train_mix)
        data_val_mix = np.concatenate(data_val_mix)
        data_test_mix = np.concatenate(data_test_mix)
        print('\n------------------- Mixed data -------------------')
        print('data train mix: ', len(data_train_mix))
        print('data val mix: ', len(data_val_mix))
        print('data test mix: ', len(data_test_mix))

        # Shuffle
        for _ in range(5):
            np.random.shuffle(data_train_mix)
            np.random.shuffle(data_val_mix)
            np.random.shuffle(data_test_mix)
        
        # Save
        # get current script directory
        save_dir = os.path.dirname(os.path.realpath(__file__))
        # get parent directory
        save_dir = os.path.dirname(save_dir)
        # cd to data folder
        save_dir = os.path.join(save_dir, 'data', 'mixed_data')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(os.path.join(save_dir, f'data_train_mix_{len(data_train_mix)}.npy'), data_train_mix, allow_pickle=True)
        np.save(os.path.join(save_dir, f'data_val_mix_{len(data_val_mix)}.npy'), data_val_mix, allow_pickle=True)
        np.save(os.path.join(save_dir, f'data_test_mix_{len(data_test_mix)}.npy'), data_test_mix, allow_pickle=True)
        print(f'Saved mixed data to {save_dir}')
        self.save_info_txt(save_dir, data_train_mix, data_val_mix, data_test_mix)

    def save_info_txt(self, save_dir, data_train_mix, data_val_mix, data_test_mix):
        # make info.txt file
        info_file = open(os.path.join(save_dir, 'info.txt'), 'w')
        info_file.write('----- Data mix info -----\n')
        info_file.write(f'{self.data_info} \n')
        info_file.write('Data mix:\n')
        info_file.write(f'  Training data     : {len(data_train_mix)}\n')
        info_file.write(f'  Validation data   : {len(data_val_mix)}\n')
        info_file.write(f'  Testing data      : {len(data_test_mix)}\n')
        info_file.write('Data folders:\n')
        info_file.close()


def main():

    nae_data_path = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_prediction_ws/src/nae/data/nae_paper_dataset/new_data_format/'
    data_info = [
        ('bamboo', nae_data_path + 'bamboo'),
        ('banana', nae_data_path + 'banana'),
        ('bottle', nae_data_path + 'bottle'),
        ('gourd', nae_data_path + 'gourd'),
        ('green', nae_data_path + 'green'),
        ('paige', nae_data_path + 'paige'),
        ('frisbee-pbl', '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_prediction_ws/src/nae/data/new_dataset_no_orientation/frisbee/frisbee-pbl/new_data_format/frisbee-pbl'),
        ('boomerang', '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_prediction_ws/src/nae/data/new_dataset_no_orientation/boomerang/min_len_65/new_data_format/boomerang'),
        ('big_plane', '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_prediction_ws/src/nae/data/new_dataset_no_orientation/big_plane/min_len_65/new_data_format/big_plane'),
    ]
    object_name_list = ['frisbee-pbl', 'big-plane-1']
    general_data_maker = GeneralDataMaker(data_info)
    general_data_maker.mix_data(object_name_list)

# call main function
if __name__ == '__main__':
    main()