import numpy as np
import os
from nae_core.utils.submodules.training_utils.data_loader import DataLoader
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
        # only mix data from the objects in the list
        # check if objects in object_name_list are in self.data_info
        for object_name in object_name_list:
            if object_name not in [x[0] for x in self.data_info]:
                raise ValueError(f'Object name {object_name} is not in the data')
        self.util_printer.print_green('All objects are in the data. Starting to mix ...', background=True)
        data_train_mix = []
        data_val_mix = []
        data_test_mix = []
        # create default dict selected_data
        selected_data = {}
        self.util_printer.print_green('Check the following info before saving mixed data')
        print(f'Loading data from {len(self.data_info)} folders')
        for idx, (object_name, data_folder) in enumerate(self.data_info):
            if object_name not in object_name_list:
                continue
            selected_data[object_name] = data_folder
            self.util_printer.print_blue(f'\n------------------- [{idx} - {object_name}] -------------------', background=True)
            print('     path: ', data_folder)

            if not os.path.exists(data_folder):
                raise FileNotFoundError(f"Folder {data_folder} does not exist")
            loader = DataLoader()
            data_train, data_val, data_test = loader.load_dataset(data_folder)
            print(f'Found:')
            print(f'     Training data     : {len(data_train)}')
            print(f'     Validation data   : {len(data_val)}')
            print(f'     Testing data      : {len(data_test)}')

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
        input('Press Enter to save mixed data ...')

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
        # synthetic all selected data name to create a new folder
        selected_data_name = '-'.join([x for x in selected_data.keys()])
        save_dir = os.path.join(save_dir, 'data', f'mixed_data-{selected_data_name}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(os.path.join(save_dir, f'data_train_mix_{len(data_train_mix)}.npy'), data_train_mix, allow_pickle=True)
        np.save(os.path.join(save_dir, f'data_val_mix_{len(data_val_mix)}.npy'), data_val_mix, allow_pickle=True)
        np.save(os.path.join(save_dir, f'data_test_mix_{len(data_test_mix)}.npy'), data_test_mix, allow_pickle=True)
        print(f'Saved mixed data to {save_dir}')
        self.save_info_txt(save_dir, selected_data, data_train_mix, data_val_mix, data_test_mix)

    def save_info_txt(self, save_dir, selected_data, data_train_mix, data_val_mix, data_test_mix):
        # make info.txt file
        info_file = open(os.path.join(save_dir, 'info.txt'), 'w')
        info_file.write('----- Data mix info -----\n')
        info_file.write(f'{selected_data} \n')
        info_file.write('Data mix:\n')
        info_file.write(f'  Training data     : {len(data_train_mix)}\n')
        info_file.write(f'  Validation data   : {len(data_val_mix)}\n')
        info_file.write(f'  Testing data      : {len(data_test_mix)}\n')
        info_file.write('Data folders:\n')
        info_file.close()


def main():

    nae_data_path = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/data/nae_paper_dataset/new_data_format'
    our_data_path = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/dynamic_nae/nae_core/data/new_dataset_no_orientation/new_format/3_enrichment'
    data_info = [
        ('bamboo', nae_data_path + '/bamboo/split/bamboo'),
        ('banana', nae_data_path + '/banana/split/banana'),
        ('bottle', nae_data_path + '/bottle/split/bottle'),
        ('gourd', nae_data_path + '/gourd/split/gourd'),
        ('green', nae_data_path + '/green/split/green'),
        ('paige', nae_data_path + '/paige/split/paige'),
        ('big_plane', our_data_path + '/big_plane/split'),
        ('frisbee', our_data_path + '/frisbee/split'),
        ('boomerang', our_data_path + '/boomerang/split'),
    ]
    object_name_list = ['bamboo', 'gourd', 'green', 'frisbee', 'boomerang']
    general_data_maker = GeneralDataMaker(data_info)
    general_data_maker.mix_data(object_name_list)

# call main function
if __name__ == '__main__':
    main()