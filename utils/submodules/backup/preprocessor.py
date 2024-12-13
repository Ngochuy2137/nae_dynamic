import numpy as np
from nae_core.utils.submodules.training_utils.data_loader import DataLoader as NAEDataLoader
import os
from datetime import datetime

class Preprocessor:
    def __init__(self) -> None:
        self.dl = NAEDataLoader()

    # Hàm để trích xuất các đoạn con có độ dài đúng bằng 50
    def extract_segments(self, traj, segment_length=50):
        segments = []
        for start in range(0, len(traj) - segment_length + 1, segment_length):
            segments.append(traj[start:start + segment_length])
        return segments

    def preprocess_rigid_data(self, data_np_org, 
                                seq_length=0, prediction_steps=0, 
                                check_vel_correctness=False, delta_t=0.0166666667, 
                                cutoff_some_first_points=False, num_first_points_to_cut=1,
                                filter_boundup=False,
                                recycle_data=False,
                                embedding = None):
        
        print('[PREPROCESSOR] Start preprocessing data')
        print('     -------------------------------')
        print('     Some notes: ')
        print('     - Units must be in cm and cm/s')
        print('     - Data for preprocessing must have axis y (not z) as up axis, respectively, the up velocity must be vy')
        print('     - The final datum of a data point must be g')
        print('     - velocities must be calculated in advanced')
        print('     -------------------------------')

        print('\n     Preprocess result: ')
    
        p_data = data_np_org.copy()
        # 1. convert from cm to m except g
        # if convert_to_meter_except_g:
        temp_data = []
        for traj in p_data:
            traj = np.array(traj)
            traj[:, :-1] = traj[:, :-1]/100
            # traj[:, -1] *= 100
            temp_data.append(traj)
        p_data = temp_data

        # 2. Swap y and z before interpolating velocity
        # if swap_yz_vy_vz:
        temp_data = []
        for traj in p_data:
            traj = np.array(traj)
            traj[:, [1, 2]] = traj[:, [2, 1]]
            traj[:, [4, 5]] = traj[:, [5, 4]]
            temp_data.append(traj)
        p_data = temp_data

        # 3. cut off the first data of each trajectory
        if cutoff_some_first_points:
            temp_data = []
            for traj in p_data:
                traj = traj[num_first_points_to_cut:]
                temp_data.append(traj)
            p_data = temp_data

        # 4. check velocity correctness
        if check_vel_correctness:
            print('     Checking velocity correctness ...')
            temp_data = []
            for traj in p_data:
                traj = np.array(traj)
                for i in range(1, 10):
                    vx = (traj[i, 0] - traj[i-1, 0]) / delta_t
                    vy = (traj[i, 1] - traj[i-1, 1]) / delta_t
                    vz = (traj[i, 2] - traj[i-1, 2]) / delta_t
                    correct = np.allclose([vx, vy, vz], [traj[i, 3], traj[i, 4], traj[i, 5]], atol=1e-4)
                    if not correct:
                        print('     Incorrect velocities: ')
                        print('     ', [vx, vy, vz], ' vs ', [traj[i, 3], traj[i, 4], traj[i, 5]])
                        raise ValueError('Velocity was not calculated correctly')
        print('         -> Velocities are correct')
        # 5. filter bound up, 
        # it is implemented only when velocities are correct
        if filter_boundup:
            prep_data = []
            for trajectory in p_data:
                filtered_trajectory = []
                # filter bound up
                filtered_trajectory.append(trajectory[0])
                for i in range(1, len(trajectory)-1):
                    if trajectory[i][5]<0 and trajectory[i+1][5]>0:
                        # stop appending when the object bound up
                        break
                    filtered_trajectory.append(trajectory[i])
                prep_data.append(filtered_trajectory)
            # prep_data = np.array(prep_data)
            p_data = prep_data
        
        # 6. recycle data
        if recycle_data:
            org_data_num = len(p_data)
            recycled_data = []
            for trajectory in p_data:
                segments = self.extract_segments(trajectory, seq_length + prediction_steps)
                if len(segments) > 1:
                    # the trajectory was separated into segments
                    recycled_data.extend(segments)
                else:
                    # else, the trajectory was not separated and preserved original data
                    recycled_data.append(trajectory)
            p_data = np.array(recycled_data, dtype=object)
            print('     Recycled ', len(p_data)-org_data_num, ' data')

        # 7. Only keep the data that has length >= seq_length + prediction_steps
        # Filter too short trajectories
        # And shorten all trajectories to seq_length + prediction_steps
        org_data_num = len(p_data)
        org_len_list = [len(d) for d in p_data]
        temp_data = [traj[:seq_length + prediction_steps] for traj in p_data if len(traj) >= seq_length + prediction_steps]
        p_data = np.array(temp_data)

        # 8. Add id to each trajectory point
        if embedding:
            temp_data = []
            for traj in p_data:
                temp_traj = []
                for p in traj:
                    p = np.concatenate((p, embedding))
                    temp_traj.append(p)
                temp_data.append(temp_traj)
            p_data = temp_data
        print('     Filtered ', org_data_num - len(p_data), ' too short trajectories. Min len was ', np.min(org_len_list))
        
        print("     Filtered ", len(data_np_org) - len(p_data), " unavailable trajectories")
        print('     Total data after preprocessing: ', p_data.shape)
        return p_data
    
    def preprocess_deformable_data(self, data_np_org,
                                    recycle_data=False,
                                    seq_length=0, prediction_steps=0, 
                                    check_vel_correctness=False, delta_t=1/60, 
                                    cutoff_some_first_points=False, num_first_points_to_cut=1,
                                    embedding = None):
        
        p_data = data_np_org.copy()
        # 1. convert from cm to m except g
        # if convert_to_meter_except_g:
        temp_data = []
        for traj in p_data:
            traj = np.array(traj)
            traj[:, :-1] = traj[:, :-1]/100
            # traj[:, -1] *= 100
            temp_data.append(traj)
        p_data = temp_data

        # 2. Swap y and z before interpolating velocity
        # if swap_yz_vy_vz:
        temp_data = []
        for traj in p_data:
            traj = np.array(traj)
            traj[:, [1, 2]] = traj[:, [2, 1]]
            traj[:, [4, 5]] = traj[:, [5, 4]]
            temp_data.append(traj)
        p_data = temp_data

        # 3. cut off the first data of each trajectory
        if cutoff_some_first_points:
            temp_data = []
            for traj in p_data:
                traj = traj[num_first_points_to_cut:]
                temp_data.append(traj)
            p_data = temp_data

        # 4. check velocity correctness
        if check_vel_correctness:
            print('     Checking velocity correctness ...')
            temp_data = []
            for traj in p_data:
                traj = np.array(traj)
                for i in range(1, 10):
                    vx = (traj[i, 0] - traj[i-1, 0]) / delta_t
                    vy = (traj[i, 1] - traj[i-1, 1]) / delta_t
                    vz = (traj[i, 2] - traj[i-1, 2]) / delta_t
                    correct = np.allclose([vx, vy, vz], [traj[i, 3], traj[i, 4], traj[i, 5]], atol=1e-4)
                    if not correct:
                        print('     Incorrect velocities: ')
                        print('     ', [vx, vy, vz], ' vs ', [traj[i, 3], traj[i, 4], traj[i, 5]])
                        raise ValueError('Velocity was not calculated correctly')
        print('         -> Velocities are correct')

        # 6. recycle data
        if recycle_data:
            org_data_num = len(p_data)
            recycled_data = []
            for trajectory in p_data:
                segments = self.extract_segments(trajectory, seq_length + prediction_steps)
                if len(segments) > 1:
                    # the trajectory was separated into segments
                    recycled_data.extend(segments)
                else:
                    # else, the trajectory was not separated and preserved original data
                    recycled_data.append(trajectory)
            p_data = np.array(recycled_data, dtype=object)
            print('     Recycled ', len(p_data)-org_data_num, ' data')

        # 7. Only keep the data that has length >= seq_length + prediction_steps
        # Filter too short trajectories
        # And shorten all trajectories to seq_length + prediction_steps
        org_data_num = len(p_data)
        org_len_list = [len(d) for d in p_data]
        temp_data = [traj[:seq_length + prediction_steps] for traj in p_data if len(traj) >= seq_length + prediction_steps]
        p_data = np.array(temp_data)

        # 8. Add id to each trajectory point
        if embedding:
            temp_data = []
            for traj in p_data:
                temp_traj = []
                for p in traj:
                    p = np.concatenate((p, embedding))
                    temp_traj.append(p)
                temp_data.append(temp_traj)
            p_data = temp_data

        print('     Filtered ', org_data_num - len(p_data), ' too short trajectories. Min len was ', np.min(org_len_list))
        print("     Filtered ", len(data_np_org) - len(p_data), " unavailable trajectories")
        print('     Total data after preprocessing: ', p_data.shape)
        return p_data
    
    '''
    Preprocess data for NAE paper dataset
    - Interpolate velocity
    - generate data with format: x, y, z, vx, vy, vz, 0, 0, g
    '''
    def preprocess_nae_paper_data(self, dataset, 
                                    swap_yz = False,
                                    num_first_points_to_cut=0):
        
        print('[PREPROCESSOR] Start preprocessing data')
        print('     -------------------------------')
        print('     Some notes: ')
        print('     - Dataset is a dictionary with keys: position, time_step')
        print('     - Data for preprocessing must have axis y (not z) as up axis, respectively')
        print('     -------------------------------')

        # 1. Swap y and z before interpolating velocity
        print('check swap 1: ', dataset[0]['position'][21])
        if swap_yz:
            temp_data = []
            for traj in dataset:
                traj_p = np.array(traj['position'])
                traj_p[:, [1, 2]] = traj_p[:, [2, 1]]
                traj['position'] = traj_p
        print('check swap 0: ', dataset[0]['position'][21])
        
            #     temp_data.append(traj)
            # dataset = temp_data

        # 1. Interpolate velocity
        prep_dataset = []
        for traj in dataset:
            one_traj = [] 
            traj_p = traj['position']
            traj_t = traj['time_step']
            if len(traj_p) != len(traj_t):
                raise ValueError('Position and time step must have the same length')
            
            for i in range(len(traj_p)):
                one_point = []
                time = traj_t[i]
                # calculate 9 values for each data point
                # - Position
                x = traj_p[i][0]
                y = traj_p[i][1]
                z = traj_p[i][2]
                # - Velocity
                if i == 0:
                    vx = 0
                    vy = 0
                    vz = 0
                    prv_time = 0
                else:
                    prv_pnt = one_traj[i-1]
                    prv_time = traj_t[i-1]
                    delta_t = time - prv_time
                    vx = (x - prv_pnt[0]) / delta_t
                    vy = (y - prv_pnt[1]) / delta_t
                    vz = (z - prv_pnt[2]) / delta_t

                # - Acceleration
                ax = 0
                ay = 0
                az = 9.81

                one_point = [x, y, z, vx, vy, vz, ax, ay, az]
                one_traj.append(one_point)
            prep_dataset.append(np.array(one_traj))

        # 2. cut off the first data of each trajectory
        if num_first_points_to_cut > 0:
            temp_data = []
            for traj in prep_dataset:
                traj = traj[num_first_points_to_cut:]
                temp_data.append(traj)
            prep_dataset = temp_data
        # convert to numpy array
        prep_dataset = np.array(prep_dataset, dtype=object)
        print('     Total data after preprocessing: ', prep_dataset.shape)
        return prep_dataset
    
    def generate_input_seqs_and_labels(self, data, seq_length, prediction_steps):
        pass

    def merge_data_from_files(self, data_path_list, thrown_object=None, delta_t=None, shuffle_data = False, enable_save=False):
        data_list = []
        count = 0
        for data_path in data_path_list:
            count += 1
            data_i = self.dl.load_dataset_from_file(data_path, title='Data ' + str(count))
            data_list.append(data_i)
        self.data = np.concatenate(data_list, axis=0)
        # Shuffle data
        if shuffle_data:
            np.random.shuffle(self.data)

        print('\n   ---------- CHECK MERGED DATA ----------')
        len_list = [len(d) for d in self.data]   
        len_list_np = np.array(len_list)
        max_len = np.max(len_list_np)
        min_len = np.min(len_list_np)
        print("     Num samples:  ", len(self.data))
        print("     Max length:   ", max_len)
        print("     Min length:   ", min_len)
        print("     Mean length:  ", np.mean(len_list_np))
        print('     check data: ', self.data[0][1])

        if enable_save:
            if thrown_object is None or delta_t is None:
                selection = input('     WARNING: thrown_object or delta_t is None. Do you want to continue? (y/n)')   
                if selection == 'n':
                    print('     Data was not saved')
                    return self.data
                thrown_object = 'unknown_object'
                delta_t = 'unknown_delta_t'
            self.dl.save_dataset(self.data, thrown_object, title='merged_data', max_len=max_len, min_len=min_len, delta_t=delta_t)
            print('     Data was saved successfully')
        return self.data
    
    def merge_data(self, data_list, thrown_object=None, delta_t=None, shuffle_data = False, enable_save=False):
        print('[PREPROCESSOR] Start merging data from: ', [len(d) for d in data_list])
        data = np.concatenate(data_list, axis=0)
        # Shuffle data
        if shuffle_data:
            np.random.shuffle(data)

        print('\n   ---------- CHECK MERGED DATA ----------')
        len_list = [len(d) for d in data]   
        len_list_np = np.array(len_list)
        max_len = np.max(len_list_np)
        min_len = np.min(len_list_np)
        print("     Num samples:  ", len(data))
        print("     Max length:   ", max_len)
        print("     Min length:   ", min_len)
        print("     Mean length:  ", np.mean(len_list_np))
        print('     check data: ', data[0][1])

        if enable_save:
            if thrown_object is None or delta_t is None:
                selection = input('     WARNING: thrown_object or delta_t is None. Do you want to continue? (y/n)')   
                if selection == 'n':
                    print('     Data was not saved')
                    return data
                thrown_object = 'unknown_object'
                delta_t = 'unknown_delta_t'
            self.dl.save_dataset(data, thrown_object, title='merged_data', max_len=max_len, min_len=min_len, delta_t=delta_t)
            print('     Data was saved successfully')
        return data
    
    def separate_and_preprocess_data(self,
                                    train_num, val_num,
                                    data_path, 
                                    recycle_data, 
                                    seq_length, prediction_steps, 
                                    check_vel_correctness, delta_t,
                                    cutoff_some_first_points, num_first_points_to_cut,
                                    filter_boundup,
                                    output_data_dir, data_type='',
                                    preprocess_mode='',
                                    embedding = None,
                                    shuffle_data=True):
        print('\n\n[PREPROCESSOR] Start separating and preprocessing: ', data_type, ' data', ' - preprocess mode: ', preprocess_mode)
        # print all parameters
        print('     Parameters: ')
        print('     - Train num: ', train_num)
        print('     - Validation num: ', val_num)
        print('     - Data path: ', data_path)
        print('     - Recycle data: ', recycle_data)
        print('     - Sequence length: ', seq_length)
        print('     - Prediction steps: ', prediction_steps)
        print('     - Check velocity correctness: ', check_vel_correctness)
        print('     - Delta t: ', delta_t)
        print('     - Cutoff some first points: ', cutoff_some_first_points)
        print('     - Number of first points to cut: ', num_first_points_to_cut)
        print('     - Filter bound up: ', filter_boundup)
        print('     - Output data dir: ', output_data_dir)
        print('     - Data type: ', data_type)
        print('     - Preprocess mode: ', preprocess_mode)
        print('     - Embedding: ', embedding)
        print('     - Shuffle data: ', shuffle_data)
        input('Are you agree to start proprocess data ?')
        # --- 1. Load data ---
        data = self.dl.load_dataset_from_file(data_path, title=data_type)

        # --- 2. Preprocess all data ---
        if preprocess_mode == 'rigid-preprocess':
            filtered_data = self.preprocess_rigid_data(data,
                                                            recycle_data=recycle_data, 
                                                            seq_length=seq_length, prediction_steps=prediction_steps,
                                                            check_vel_correctness=check_vel_correctness, delta_t=delta_t,
                                                            cutoff_some_first_points = cutoff_some_first_points, num_first_points_to_cut = num_first_points_to_cut,
                                                            filter_boundup = filter_boundup,
                                                            embedding = embedding)
        elif preprocess_mode == 'deformable-preprocess':
            filtered_data = self.preprocess_deformable_data(data,
                                                            recycle_data = recycle_data,
                                                            seq_length=seq_length, prediction_steps=prediction_steps,
                                                            check_vel_correctness = check_vel_correctness, delta_t=delta_t,
                                                            cutoff_some_first_points = cutoff_some_first_points, num_first_points_to_cut = num_first_points_to_cut,
                                                            embedding = embedding)
        else:
            raise ValueError('Preprocess mode is not valid')
        
        # check a trajectory after preprocessing
        import random
        checked_traj = filtered_data[random.randint(0, len(filtered_data)-1)]
        print('Check data after preprocessing: ')
        for p in checked_traj:
            print(p)

        # --- 3. Shuffle data ---
        if shuffle_data:
            np.random.shuffle(data)

        # --- 4. Separate data into training, validation, testing data ---
        input('Press ENTER to separete data ...')
        if train_num + val_num > len(filtered_data):
            raise ValueError('train_num + val_num must be less than or equal to the length of data')
        
        training_data = filtered_data[:train_num]
        val_data = filtered_data[train_num:train_num+val_num]
        testing_data = filtered_data[train_num+val_num:]
        print('     Training data: ', len(training_data))
        print('     Validation data: ', len(val_data))
        print('     Testing data: ', len(testing_data))


        # --- 3. Save preprocessed data ---
        # save preprocessed separated data
        print('\nSaving preprocessed data ...')
        proc_data_dict = {
            'training': training_data,
            'validate': val_data,
            'testing': testing_data
        }

        for key, sep_data in proc_data_dict.items():
            data_org_dir = os.path.join(output_data_dir, data_type + '_' + str(sep_data.shape[1]) + '_' + preprocess_mode)
            os.makedirs(data_org_dir, exist_ok=True)
            data_path = os.path.join(data_org_dir, f'{key}_data_' + 'len_' + str(len(sep_data)) + '.npy')
            np.save(data_path, sep_data)
            print('     preprocessed', key, ' data: were saved successfully at ', data_path)


        # Create info file
        # data_info_dir = os.path.join(output_data_dir)
        info_file = os.path.join(data_org_dir, 'info.txt')
        with open(info_file, 'w') as f:
            f.write(f"Data type: {data_type}\n")
            f.write(f"Preprocess mode: {preprocess_mode}\n")
            f.write(f"Recycle data: {recycle_data}\n")
            f.write(f"Interpolate velocity: {check_vel_correctness}\n")
            f.write(f"Delta t: {delta_t}\n")
            f.write(f"Cutoff some first points: {cutoff_some_first_points}\n")
            f.write(f"Number of first points to cut: {num_first_points_to_cut}\n")
            f.write(f"Filter bound up: {filter_boundup}\n")
            f.write(f"Sequence length: {seq_length}\n")
            f.write(f"Prediction steps: {prediction_steps}\n")
        print('     Info file was saved successfully at ', info_file)

        return training_data, val_data, testing_data
    
    def split_and_preprocess_data(self,
                                train_ratio, val_ratio,
                                data_path, 
                                num_first_points_to_cut,
                                output_data_dir, data_type='',
                                shuffle_data=True):
        # print('\n\n[PREPROCESSOR] Start separating and preprocessing: ', data_type, ' data', ' - preprocess mode: ', preprocess_mode)
        print('     Parameters: ')
        print('     - Training percentage: ', train_ratio)
        print('     - Validation percentage: ', val_ratio)
        print('     - Data path: ', data_path)
        print('     - Number of first points to cut: ', num_first_points_to_cut)
        print('     - Output data dir: ', output_data_dir)
        print('     - Data type: ', data_type)
        print('     - Shuffle data: ', shuffle_data)
        input('Are you agree to start proprocess data ?')
        # --- 1. Load data ---
        dataset = self.dl.load_dataset_from_folder(data_path, file_format='npz')

        # --- 2. Preprocess dataset ---
        prep_data = self.preprocess_nae_paper_data(dataset, swap_yz = True,
                                                    num_first_points_to_cut = num_first_points_to_cut)
        
        # # check a trajectory after preprocessing
        # import random
        # checked_traj = prep_data[random.randint(0, len(prep_data)-1)]
        # print('Check data after preprocessing: ')
        # for p in checked_traj:
        #     print('     ', p)

        # --- 3. Split data into training, validation, testing data ---
        input('Ready to split the data into training dataset, validation dataset, testing dataset. \nPress ENTER to split the data ...')
        # Shuffle data
        if shuffle_data:
            np.random.shuffle(prep_data)
        # Split dataset
        train_num = int(len(prep_data) * train_ratio)
        val_num = int(len(prep_data) * val_ratio)
        
        training_data = prep_data[:train_num]
        val_data = prep_data[train_num:train_num+val_num]
        testing_data = prep_data[train_num+val_num:]
        print('     Training data: ', len(training_data))
        print('     Validation data: ', len(val_data))
        print('     Testing data: ', len(testing_data))


        # --- 4. Save datasets ---
        print('\nSaving datasets ...')
        dataset_dict = {
            'training': training_data,
            'validate': val_data,
            'testing': testing_data
        }

        for key, sep_data in dataset_dict.items():
            data_org_dir = os.path.join(output_data_dir, data_type + '_' + str(prep_data.shape[0]))
            os.makedirs(data_org_dir, exist_ok=True)
            data_path = os.path.join(data_org_dir, f'{key}_data_' + 'len_' + str(len(sep_data)) + '.npy')
            np.save(data_path, sep_data)
            print('     preprocessed', key, ' data: were saved successfully at ', data_path)


        # Calculate max, min, mean length of trajectories
        len_list = [len(d) for d in prep_data]
        len_list_np = np.array(len_list)
        max_len = np.max(len_list_np)
        min_len = np.min(len_list_np)
        mean_len = np.mean(len_list_np)
        print('     Max length: ', max_len)
        print('     Min length: ', min_len)
        print('     Mean length: ', mean_len)
        # Create info file
        # data_info_dir = os.path.join(output_data_dir)
        info_file = os.path.join(data_org_dir, 'info.txt')
        with open(info_file, 'w') as f:
            f.write(f"Data type: {data_type}\n")
            f.write(f"Number of first points to cut: {num_first_points_to_cut}\n")
            f.write(f"Trajectories length:\n")
            f.write(f"     Max length: {max_len}\n")
            f.write(f"     Min length: {min_len}\n")
            f.write(f"     Mean length: {mean_len}\n")
        print('     Info file was saved successfully at ', info_file)

        return training_data, val_data, testing_data
    
    
    '''
    Create general data for training. Merge data from multiple folders
    Args:
        data_folder_list: list of data folders for objects
        shuffle_data: shuffle data or not
    '''
    def create_mixed_data(self, data_folder_list, shuffle_data=False):
        data_train = []
        data_val = []
        data_test = []
        for data_folder in data_folder_list:
            data_train_i, data_val_i, data_test_i = self.dl.load_dataset(data_folder)
            data_train.append(data_train_i)
            data_val.append(data_val_i)
            data_test.append(data_test_i)
        data_train = np.concatenate(data_train, axis=0)
        data_val = np.concatenate(data_val, axis=0)
        data_test = np.concatenate(data_test, axis=0)

        if shuffle_data:
            np.random.shuffle(data_train)
            np.random.shuffle(data_val)
            np.random.shuffle(data_test)
        return data_train, data_val, data_test