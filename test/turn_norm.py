import os
import nibabel as nib
import numpy as np

def max_min_normalization(data):
    """
    对输入的数据进行最大 - 最小归一化处理
    :param data: 输入的 NumPy 数组
    :return: 归一化后的 NumPy 数组
    """
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val - min_val == 0:
        return np.zeros_like(data)
    return (data - min_val) / (max_val - min_val)

def convert_nii_to_npy(input_folder, output_folder):
    """
    将输入文件夹中的所有 nii.gz 文件转换为 NumPy 数组，并进行最大 - 最小归一化处理
    :param input_folder: 包含 nii.gz 文件的输入文件夹路径
    :param output_folder: 保存归一化后 NumPy 数组的输出文件夹路径
    """
    # 如果输出文件夹不存在，则创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.nii.gz'):
            # 构建完整的文件路径
            file_path = os.path.join(input_folder, filename)
            # 读取 nii.gz 文件
            nii_img = nib.load(file_path)
            # 获取图像数据
            data = nii_img.get_fdata()
            # 进行最大 - 最小归一化处理
            normalized_data = max_min_normalization(data)
            # 构建输出文件名
            output_filename = os.path.splitext(os.path.splitext(filename)[0])[0] + '.npy'
            output_path = os.path.join(output_folder, output_filename)
            # 保存归一化后的 NumPy 数组
            np.save(output_path, normalized_data)
            print(f"Processed {filename} and saved to {output_path}")

if __name__ == "__main__":
    input_folder = "../data/DTI_original"
    output_folder = "../data/DTI"
    convert_nii_to_npy(input_folder, output_folder)