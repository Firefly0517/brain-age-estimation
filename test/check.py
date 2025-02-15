import os
import pandas as pd

def remove_orphan_and_empty_age_files(folder1, folder2, folder3):

    # 获取两个文件夹中的文件名列表
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))
    files3 = set(os.listdir(folder3))

    # 找出只在其中一个文件夹中存在的文件名
    unique_files1 = files1 - files2
    unique_files2 = files2 - files1
    unique_files3 = files1 - files3
    unique_files4 = files3 - files1

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

    for file in unique_files3:
        file_path1 = os.path.join(folder1, file)
        file_path2 = os.path.join(folder2, file)
        if os.path.isfile(file_path1):
            os.remove(file_path1)
            print(f"Deleted {file_path1}")
        if os.path.isfile(file_path2):
            os.remove(file_path2)
            print(f"Deleted {file_path2}")

    for file in unique_files4:
        file_path = os.path.join(folder3, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted {file_path}")

if __name__ == "__main__":
    folder1 = "../data/T1"
    folder2 = "../data/DTI"
    table_path = "../data/Age"
    remove_orphan_and_empty_age_files(folder1, folder2, table_path)