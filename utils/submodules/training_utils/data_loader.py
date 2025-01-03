import numpy as np
import matplotlib.pyplot as plt
import os
import fnmatch
from datetime import datetime
from python_utils.printer import Printer
from nae_core.utils.submodules.preprocess_utils.data_raw_reader import RoCatDataRawReader

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

    def look_for_training_validation_data_test_folders(self, data_folder, 
                                                        data_train_keyword="data_train", 
                                                        data_val_keyword="data_val", 
                                                        data_test_keyword="data_test"):
            """
            Tìm các folder chứa từ khóa 'data_train', 'data_val', 'data_test' ở bất kỳ cấp nào trong thư mục data_folder.

            Args:
                data_folder (str): Thư mục gốc để tìm kiếm.
                data_train_keyword (str): Từ khóa để tìm folder chứa dữ liệu train.
                data_val_keyword (str): Từ khóa để tìm folder chứa dữ liệu validation.
                data_test_keyword (str): Từ khóa để tìm folder chứa dữ liệu test.

            Returns:
                dict: Dictionary với từ khóa làm key và đường dẫn folder phù hợp làm value.
            """
            matched_folders = {keyword: [] for keyword in [data_train_keyword, data_val_keyword, data_test_keyword]}
            keywords = [data_train_keyword, data_val_keyword, data_test_keyword]

            for root, dirs, _ in os.walk(data_folder):
                for keyword in keywords:
                    # Tìm folder chứa từ khóa
                    matched = [d for d in dirs if keyword in d]
                    
                    # Lưu các folder khớp vào matched_folders
                    for folder in matched:
                        matched_folders[keyword].append(os.path.join(root, folder))

            # Kiểm tra và xử lý cảnh báo
            for keyword in keywords:
                if len(matched_folders[keyword]) == 0:
                    self.util_printer.print_yellow(f"Warning: No folder found with keyword: {keyword}")
                    return None
                elif len(matched_folders[keyword]) > 1:
                    self.util_printer.print_yellow(f"Warning: Found more than 1 folder with keyword: {keyword}")
                    return None

            # Chỉ giữ folder đầu tiên tìm thấy cho mỗi từ khóa
            for keyword in keywords:
                matched_folders[keyword] = matched_folders[keyword][0]

            return matched_folders
    
    def load_train_val_test_dataset(self, data_folder):
        data_path_dict = self.look_for_training_validation_data_test_folders(data_folder)
        if data_path_dict is None:
            self.util_printer.print_red("No data files found")
            return None
        self.util_printer.print_green('Loading data for model. Found data files ', background=True)
        for (keyword, paths) in data_path_dict.items():
            print(f"    {keyword}: {paths}")
        self.util_printer.print_green('--------------------------------------------')

        data_train_path = data_path_dict['data_train']
        data_val_path = data_path_dict['data_val']
        data_test_path = data_path_dict['data_test']
        # load data
        data_train = RoCatDataRawReader(data_train_path).read()
        data_val = RoCatDataRawReader(data_val_path).read()
        data_test = RoCatDataRawReader(data_test_path).read()
        return data_train, data_val, data_test
    


def main():
    pass
if __name__ == '__main__':
    main()





