import numpy as np

class InputLabelGenerator:
    def __init__(self) -> None:
        pass

    '''
    Generate input and label sequences from data
    We will slide a window of size input_len + future_pred_len over each traj in data
    Args:
        data: list of trajectories
        input_len: length of input sequence
        future_pred_len: length of future prediction sequence
            -> label = [0:input_len + future_pred_len]
            -> label_len = input_len + future_pred_len
            - When training, we need to split label into labels for loss1, loss2, loss3
                - Loss1: 1 -> t
                - Loss2: t+1 -> t+k-1
                - Loss3: 0 -> t-1
        increment: increment for sliding window
        shuffle_data: shuffle the data
    '''
    def generate_input_label_seq(self, data, input_len, future_pred_len, increment=1, shuffle_data = True):
        input_seq = []
        label_seq = []
        window_size = input_len + future_pred_len

        for traj in data:
            traj_len = len(traj)
            if traj_len < window_size:
                continue
            for i in range(0, traj_len - window_size + 1, increment):
                input_seq.append(traj[i:i+input_len])
                label_seq.append(traj[i:i+input_len+future_pred_len])
        if shuffle_data:
            combined = list(zip(input_seq, label_seq))  # Ghép từng cặp (input, label) lại
            np.random.shuffle(combined)  # Shuffle các cặp
            input_seq, label_seq = zip(*combined)  # Tách các cặp ra lại thành hai danh sách
            input_seq = np.array(input_seq)
            label_seq = np.array(label_seq)
        return input_seq, label_seq
        

    def segment_data(self, data, window_size, increment=1, shuffle_data = True):
        """
        Segment data into sequences of length window_size
        """
        data_list = []
        for traj in data:
            traj_len = len(traj)
            if traj_len < window_size:
                continue
            for i in range(0, traj_len - window_size + 1, increment):
                data_list.append(traj[i:i+window_size])
        if shuffle_data:
            np.random.shuffle(data_list)
        return data_list

# Test the function
def main():
    data = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    data_segmentation = InputLabelGenerator()
    window_size = 3
    increment = 1
    shuffle_data = True
    data_list = data_segmentation.segment_data(data, window_size, increment, shuffle_data)
    print(data_list)
if __name__ == '__main__':
    main()