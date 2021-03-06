# coding=utf-8
import scipy.io as sio
import pandas as pd
import os
import numpy as np

def mat2csv():
    """
    将当前目录下的data目录下的 .mat 文件转换成多个 .csv文件
    :return:
    """
    np.set_printoptions(threshold=np.NaN) 
    curr_path = os.path.dirname(__file__)
    mat_data_path = os.path.join(curr_path, "data")
    csv_data_path = os.path.join(curr_path, "csv")
    if not os.path.exists(csv_data_path):
        os.makedirs(csv_data_path)
    if not os.path.exists(mat_data_path):
        os.makedirs(mat_data_path)

    file_list = os.listdir(mat_data_path)
    mat_list = [file_name for file_name in file_list if file_name.endswith(".mat")]
    print("find mat file : ", mat_list)

    for mat_file in mat_list:
        file_path = os.path.join(mat_data_path, mat_file)
        csv_data_fold = os.path.join(csv_data_path,mat_file.split('.')[0])
        if not os.path.exists(csv_data_fold):
            os.makedirs(csv_data_fold)
        mat_data = sio.loadmat(file_path)

        for key in mat_data:
            if not str(key).startswith("__"):
                data = mat_data[key][:]
                print(key)
                try:
                    dfdata = pd.DataFrame(data)
                except ValueError as e:
                    print(e.message)
                    continue
                csv_path = os.path.join(csv_data_fold, key+'.csv')
                dfdata.to_csv(csv_path)


if __name__ == "__main__":
    mat2csv()