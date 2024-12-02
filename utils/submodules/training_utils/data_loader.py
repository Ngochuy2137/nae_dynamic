import numpy as np
import matplotlib.pyplot as plt
import os
import fnmatch
from datetime import datetime
from python_utils.python_utils.printer import Printer

class DataLoader:
    '''
    Load training, validation, and testing data for the NAE model
    '''
    def __init__(self, ):
        self.util_printer = Printer()
    
    def look_for_training_validation_data_test_files(self, data_folder, data_train_keyword="data_train", data_val_keyword="data_val", data_test_keyword="data_test"):
        matched_files = {keyword: [] for keyword in [data_train_keyword, data_val_keyword, data_test_keyword]}
        keywords = [data_train_keyword, data_val_keyword, data_test_keyword]

        for root, _, files in os.walk(data_folder):
            for keyword in keywords:
                # Tìm các file khớp với keyword
                matched = fnmatch.filter(files, f"*{keyword}*")
                
                # Lưu các file khớp vào matched_files
                for filename in matched:
                    matched_files[keyword].append(os.path.join(root, filename))

        # some warnings
        for keyword in keywords:
            if len(matched_files[keyword]) == 0:
                self.util_printer.print_yellow(f"Warning: No file found with keyword: {keyword}")
                return None
            elif len(matched_files[keyword]) > 1:
                self.util_printer.print_yellow(f"Warning: Found more than 1 file with keyword: {keyword}")
                return None

        # remove the list of files for each keyword
        for keyword in keywords:
            matched_files[keyword] = matched_files[keyword][0]
        
        return matched_files

    def load_dataset(self, data_folder):
        data_path_dict = self.look_for_training_validation_data_test_files(data_folder)
        if data_path_dict is None:
            self.util_printer.print_red("No data files found")
            return None
        print('Found data files ', len(data_path_dict))
        
        for (keyword, paths) in data_path_dict.items():
            print(f"    {keyword}: {paths}")

        data_train_path = data_path_dict['data_train']
        data_val_path = data_path_dict['data_val']
        data_test_path = data_path_dict['data_test']
        data_train = np.load(data_train_path, allow_pickle=True)
        data_val = np.load(data_val_path, allow_pickle=True)
        data_test = np.load(data_test_path, allow_pickle=True)
        return data_train, data_val, data_test
    


def main():
    pass
if __name__ == '__main__':
    main()





