from nae.utils.submodules.backup.preprocessor import Preprocessor
import numpy as np
import os
import time

def main():
    data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae/data/nae_paper_dataset/split/2024-09-27/'
    data_folder_list = [data_dir + 'banana', 
                        data_dir + 'bamboo', 
                        data_dir + 'bottle',
                        data_dir + 'gourd', 
                        data_dir + 'green',
                        data_dir + 'paige']
    pp = Preprocessor()
    data_train, data_val, data_test = pp.create_general_data(data_folder_list, shuffle_data=True)

    # save data
    
    # create data folder named general_data at the same folder as this script if not exist
    mixed_data_folder = 'general_data'
    cur_time = time.strftime("%Y-%m-%d")
    data_dir = os.path.join(os.path.dirname(__file__), mixed_data_folder, cur_time)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    np.save(data_dir + '/data_train_len_' + str(len(data_train)) + '.npy', data_train)
    np.save(data_dir + '/data_val_len_' + str(len(data_val)) + '.npy', data_val)
    np.save(data_dir + '/data_test_len_' + str(len(data_test)) + '.npy', data_test)
    print('Data saved at: ', data_dir)

# call main function
if __name__ == '__main__':
    main()