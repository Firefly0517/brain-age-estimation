import pandas as pd
import numpy as np
import os

def save_age_as_npy(csv_file_path):
    # 读取 Excel 文件
    # df = pd.read_excel(excel_file_path, engine='openpyxl')
    df = pd.read_csv(csv_file_path)

    # 确保输出目录存在
    output_dir_Age = '../ADNI/Age'
    output_dir_Gender = '../ADNI/Gender'

    if not os.path.exists(output_dir_Age):
        os.makedirs(output_dir_Age)

    if not os.path.exists(output_dir_Gender):
        os.makedirs(output_dir_Gender)

    # 遍历每一行数据
    for index, row in df.iterrows():
        age = row['Age']
        gender = row['Sex']
        subject = row['Subject']
        acq_date = pd.to_datetime(row['Acq Date']).strftime('%Y-%m-%d')
        ADNI_id = str(row['Subject'] + row['Acq Date'])
        file_name = f'{subject}_{acq_date}.npy'
        file_path_Age = os.path.join(output_dir_Age, file_name)
        file_path_Gender = os.path.join(output_dir_Gender, file_name)

        if gender == 'M':
            gender = 1
        elif gender == 'F':
            gender = 0
        else:
            print(f"未知的性别值: {gender}")

        # 将 AGE 和 Gender 数据保存为 .npy 文件
        np.save(file_path_Age, np.array([age]))
        np.save(file_path_Gender, np.array([gender]))
        print(f'Saved {file_path_Age} and {file_path_Gender}')

if __name__ == "__main__":
    csv_file_path = r'E:\wechat_file\WeChat Files\wxid_fmynfmungatq22\FileStorage\File\2025-03\2.4_3_17_2025.csv'  # 替换为实际的 Excel 文件路径
    save_age_as_npy(csv_file_path)