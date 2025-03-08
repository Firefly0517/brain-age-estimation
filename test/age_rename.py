import numpy as np
import os

def rename_and_save_npy(file_path, output_dir):
    """ 读取 .npy 文件并以新名称保存 """
    # 读取数据
    data = np.load(file_path)

    # 获取原文件名（不带后缀）
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 定义新的保存路径
    save_npy_path = os.path.join(output_dir, f"{file_name}_3.npy")

    # 保存新文件
    np.save(save_npy_path, data)

    print(f"Renamed and saved: {file_path} -> {save_npy_path}")

def batch_rename_npy(input_dir, output_dir):
    """ 批量重命名并保存 .npy 文件 """
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".npy"):
            file_path = os.path.join(input_dir, file_name)
            rename_and_save_npy(file_path, output_dir)

if __name__ == "__main__":
    input_dir = r"E:\脑龄\IXI\Age"  # 存储年龄的 .npy 文件目录
    output_dir = r"E:\脑龄\IXI_2d\Age"  # 输出目录

    batch_rename_npy(input_dir, output_dir)
