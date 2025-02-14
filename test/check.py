import os
import pandas as pd

def remove_orphan_and_empty_age_files(folder1, folder2, table_path):
    # 读取 Excel 表格数据
    df = pd.read_excel(table_path)

    # 获取两个文件夹中的文件名列表
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))

    # 找出只在其中一个文件夹中存在的文件名
    unique_files1 = files1 - files2
    unique_files2 = files2 - files1

    # 删除文件夹 1 中单独存在的文件
    for file in unique_files1:
        file_path = os.path.join(folder1, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted {file_path}")

    # 删除文件夹 2 中单独存在的文件
    for file in unique_files2:
        file_path = os.path.join(folder2, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted {file_path}")

    # 删除 AGE 列为空的行
    df = df.dropna(subset=['AGE'])
    # 保存修改后的 DataFrame 回 Excel 文件
    df.to_excel("../data/age_new.xls", index=False, engine='openpyxl')

if __name__ == "__main__":
    folder1 = "../data/T1"
    folder2 = "../data/DTI"
    table_path = "../data/age.xls"
    remove_orphan_and_empty_age_files(folder1, folder2, table_path)