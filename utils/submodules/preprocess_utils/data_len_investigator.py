from nae.utils.submodules.preprocess_utils.data_raw_reader import RoCatRLLabDataRawReader, RoCatNAEDataRawReader
from nae.utils.submodules.preprocess_utils.merger import RoCatRLLabDataMerger, RoCatNAEDataMerger
from nae.utils.submodules.printer import Printer
import numpy as np
import matplotlib.pyplot as plt
import os

class RoCatRLLabDataRawInvestigator:
    def __init__(self,):
        self.util_printer = Printer()

    '''
    Check data len distribution
    '''
    def check_data_len_distribution(self, data):
        data_len = []
        for trajectory in data:
            data_len.append(len(trajectory['points']))
        data_len = np.array(data_len)
        print('Data length distribution')
        print('     - Min: ', np.min(data_len))
        print('     - Max: ', np.max(data_len))
        print('     - Mean: ', np.mean(data_len))
        print('     - Std: ', np.std(data_len))

    def plot_histogram_data_len(self, data):
        'plot 2D histogram of data length'
        data_len = [len(trajectory['points']) for trajectory in data]
        # Tạo histogram
        plt.figure(figsize=(15, 6))
        counts, bins, patches = plt.hist(data_len, bins=range(0, max(data_len) + 5, 5), 
                                        alpha=0.7, edgecolor='black', color='skyblue')
        # Tùy chỉnh trục X
        plt.xticks(np.arange(0, max(data_len) + 5, 5))
        # Thêm số lượng (frequency) lên mỗi bar
        for count, x in zip(counts, bins):
            if count > 0:  # Chỉ ghi số trên các cột có dữ liệu
                plt.text(x + 2.5, count + 0.5, str(int(count)), ha='center', fontsize=12, color='red')

        # Tùy chỉnh tiêu đề và nhãn
        plt.title("Histogram of Data Length Distribution", fontsize=16)
        plt.xlabel("Data Length", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)

        # Hiển thị biểu đồ
        plt.grid(axis='y', alpha=0.75)
        plt.show()

class RoCatNAEDataRawInvestigator:
    def __init__(self,):
        self.util_printer = Printer()
    
    '''
    Check data len distribution
    '''
    def check_data_len_distribution(self, data):
        data_len = []
        for trajectory in data:
            data_len.append(len(trajectory))
        data_len = np.array(data_len)
        print('Data length distribution')
        print('     - Min: ', np.min(data_len))
        print('     - Max: ', np.max(data_len))
        print('     - Mean: ', np.mean(data_len))
        print('     - Std: ', np.std(data_len))

    def plot_histogram_data_len(self, data):
        'plot 2D histogram of data length'
        data_len = [len(trajectory) for trajectory in data]
        # Tạo histogram
        plt.figure(figsize=(15, 6))
        counts, bins, patches = plt.hist(data_len, bins=range(0, max(data_len) + 5, 5), 
                                        alpha=0.7, edgecolor='black', color='skyblue')
        # Tùy chỉnh trục X
        plt.xticks(np.arange(0, max(data_len) + 5, 5))
        # Thêm số lượng (frequency) lên mỗi bar
        for count, x in zip(counts, bins):
            if count > 0:  # Chỉ ghi số trên các cột có dữ liệu
                plt.text(x + 2.5, count + 0.5, str(int(count)), ha='center', fontsize=12, color='red')

        # Tùy chỉnh tiêu đề và nhãn
        plt.title("Histogram of Data Length Distribution", fontsize=16)
        plt.xlabel("Data Length", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)

        # Hiển thị biểu đồ
        plt.grid(axis='y', alpha=0.75)
        plt.show()
    


def main():
    ## If you want to check Robot learning lab data:
    # data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_prediction_ws/src/nae/data/frisbee-pbl/frisbee-pbl_merged_275.npz'
    # reader = RoCatNAEDataRawReader(data_dir)
    # data_investigator = RoCatRLLabDataRawInvestigator()
    # data = reader.read()

    ## If you want to check NAE data:
    data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_prediction_ws/src/nae/data/nae_paper_dataset/split/2024-09-27/'
    object_name = 'paige'
    data_dir = os.path.join(data_dir, object_name)
    # merge data
    merger = RoCatNAEDataMerger(data_dir, object_name, file_format='npy')
    data = merger.get_data()
    data_investigator = RoCatNAEDataRawInvestigator()

    data_investigator.check_data_len_distribution(data)
    data_investigator.plot_histogram_data_len(data)

if __name__ == '__main__':
    main()