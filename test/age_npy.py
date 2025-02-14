import pandas as pd
import numpy as np
import os

def save_age_as_npy(excel_file_path):
    # 读取 Excel 文件
    df = pd.read_excel(excel_file_path, engine='openpyxl')

    # 确保输出目录存在
    output_dir = '../data/age'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历每一行数据
    for index, row in df.iterrows():
        age = row['AGE']
        ixi_id = str(int(row['IXI_ID'])).zfill(3)  # 将 IXI_ID 转换为三位字符串
        file_name = f'IXI-{ixi_id}.npy'
        file_path = os.path.join(output_dir, file_name)

        # 将 AGE 数据保存为 .npy 文件
        np.save(file_path, np.array([age]))
        print(f'Saved {file_path}')

if __name__ == "__main__":
    excel_file_path = '../data/age_new.xls'  # 替换为实际的 Excel 文件路径
    save_age_as_npy(excel_file_path)