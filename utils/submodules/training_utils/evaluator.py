from nae import *
import random
import math
from collections import defaultdict
import json
import time
import pandas as pd
DEVICE = torch.device("cuda")

class Evaluator:
    def __init__(self):
        pass

    def find_best_model_recursive(self, folder_path):
        # Kiểm tra tất cả các thư mục con
        for root, dirs, files in os.walk(folder_path):
            for dir_name in dirs:
                if dir_name.startswith('@'):  # Tìm thư mục bắt đầu bằng '@'
                    return os.path.join(root, dir_name)
        return None
    def find_best_models(self, root_folder):
        best_models = {}

        # Lặp qua các folder trong root_folder
        for folder_name in os.listdir(root_folder):
            folder_path = os.path.join(root_folder, folder_name)
            # Kiểm tra xem có phải là một thư mục không và không phải là thư mục ẩn
            if os.path.isdir(folder_path) and not folder_name.startswith('.'):
                # Tìm model tốt nhất trong tất cả các thư mục con
                best_model = self.find_best_model_recursive(folder_path)
                if best_model:
                    best_models[folder_name.split('-')[0]] = best_model
                else:
                    print('\nCannot find best model for object: ', folder_name)
                    input('Do you want to continue?')
        return best_models
    

    # Hàm để chuyển đổi tất cả giá trị float32 về float
    def convert_to_float(self, obj):
        if isinstance(obj, np.float32):  # Nếu là kiểu numpy.float32
            return float(obj)
        if isinstance(obj, dict):  # Nếu là dict, chuyển đổi đệ quy
            return {k: self.convert_to_float(v) for k, v in obj.items()}
        if isinstance(obj, list):  # Nếu là list, chuyển đổi đệ quy
            return [self.convert_to_float(i) for i in obj]
        return obj  # Trả về đối tượng nếu không cần chuyển đổi
    
    # save evaluation results to a file
    def save_results(self, scores_dict, file_path):
        scores_dict = self.convert_to_float(scores_dict)
        with open(file_path, 'w') as f:
            json.dump(scores_dict, f, indent=4)
        print("Scores have been saved to file: ", file_path)

    def save_results_to_excel(self, scores_dict, file_path):
        scores_dict = self.convert_to_float(scores_dict)  # Giả sử hàm này vẫn cần để xử lý dữ liệu
        data = []

        # Duyệt qua từng mục trong dictionary
        for score_type, objects in scores_dict.items():
            for obj, models in objects.items():
                for model, value in models.items():
                    data.append({
                        "Score Type": score_type,
                        "Object": obj,
                        "Model": model,
                        "Value": value
                    })

        # Chuyển đổi thành DataFrame
        df = pd.DataFrame(data)

        # Lưu thành file Excel với từng Score Type là một sheet riêng
        with pd.ExcelWriter(file_path) as writer:
            for score_type in df['Score Type'].unique():
                # Lọc theo từng Score Type
                filtered_df = df[df['Score Type'] == score_type]

                # Tạo bảng pivot
                pivot_df = filtered_df.pivot_table(index="Model", columns="Object", values="Value", aggfunc='first')

                # Ghi vào file Excel
                pivot_df.to_excel(writer, sheet_name=score_type)

        print("Scores have been saved to file: ", file_path)

    '''
    function evaluate
        Input:
            object_info: dict   
                - Information of the object to be evaluated:
                    + type: str     - Type of the object
                    + data: 
            model_info: dict    
                - Information of the model to be evaluated
                    + type: str     - Type of the model
                    + path: str      - Path to the directory of the model

    '''
    def evaluate(self, object_data, model_info, model_params, training_params, capture_radius, plot=False):
        np.set_printoptions(precision=4)
        nae = NAE(**model_params, **training_params, data_dir='', device=DEVICE)

        print('     ----- info -----')
        print('     Object type     :', object_data['type'])
        print('     Object data     :', len(object_data['data']['input_seqs']))
        print('     Model type      :', model_info['type'])
        print('     ----------------\n')
        # input('ENTER to continue')
        # 1. load data 
        print('    Loading data: ', object_data['type'])
        input_seqs = object_data['data']['input_seqs']
        label_seqs = object_data['data']['label_seqs'][:, 1:, :]

        # 2. load model
        print('    Loading model: ', model_info['path'])
        nae.load_model(model_info['path'])
        # ===================== PREDICT =====================
        print('PREDICTING')

        predicted_seqs = nae.predict(input_seqs, evaluation=True).cpu().numpy()
        print('     input_len       :', len(object_data['data']['input_seqs'][9]))
        print('     future_pred_len :', len(predicted_seqs[9]))

        # scoring
        mean_mse_all, mean_mse_xyz, mean_ade, \
        (mean_nade, var_nade), (mean_fe, var_fe), \
        mean_nade_future, capture_success_rate = nae.utils.score_all_predictions(predicted_seqs, 
                                                                                    label_seqs, 
                                                                                    training_params['future_pred_len'],
                                                                                    capture_thres=capture_radius)
        print('\nPrediction score:')
        print('     Object: ', object_data['type'])
        print('     Model: ', model_info['type'])
        # print(' Mean MSE all            : ', mean_mse_all)
        # print(' Mean MSE xyz            : ', mean_mse_xyz )
        # print(' Mean ADE             (m): ', mean_ade)
        # print(' Mean NADE               : ', mean_nade)
        # print(' Mean NADE future        : ', mean_nade_future)
        print('     Mean final step err (m) : ', mean_fe)
        print('     Capture success rate    : ', capture_success_rate)
        num_trials = len(object_data['data']['input_seqs'])
        print('     trials: ', num_trials)

        # plotting
        if plot:
            plot_title = 'object ' + object_data['type'] + ' - model ' + model_info['type']
            nae.utils.plotter.plot_predictions(predicted_seqs, label_seqs, lim_plot_num=5, swap_y_z=True, title=plot_title)
        
        return object_data['type'], model_info['type'], (mean_fe, var_fe), capture_success_rate