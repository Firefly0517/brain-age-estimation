import numpy as np
import os
import matplotlib.pyplot as plt

def load_npy(file_path):
    """ 加载 .npy 文件 """
    return np.load(file_path)

def slice_mri_data(data):
    """ 获取 MRI 数据的矢状切片 """
    z = data.shape[2]
    sagittal_slice = data[:, :, z // 2]  # 矢状 (Z)
    return sagittal_slice

def save_slice(slice_data, save_path):
    """ 保存切片为 .npy 文件 """
    np.save(save_path, slice_data)

def display_single_slice(slice_data, save_path):
    """ 保存单个切片的展示图像 """
    plt.figure(figsize=(5, 5))
    plt.imshow(slice_data, cmap='gray')
    plt.axis('off')
    plt.title("Sagittal Slice")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def process_and_save_slices(file_path, output_dir):
    """ 处理单个 .npy 文件并保存矢状切片（.npy 和图像） """
    # 读取数据
    mri_data = load_npy(file_path)

    # 获取矢状切片
    sagittal_slice = slice_mri_data(mri_data)

    # 获取原文件名（不带后缀）
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 定义保存路径（图像和 .npy 使用相同的前缀）
    save_image_path = os.path.join(output_dir, f"{file_name}_3.png")
    save_npy_path = os.path.join(output_dir, f"{file_name}_3.npy")

    # 保存图像
    # display_single_slice(sagittal_slice, save_image_path)

    # 保存 .npy 文件
    save_slice(sagittal_slice, save_npy_path)

    print(f"Processed: {file_path} -> Saved as {save_image_path} and {save_npy_path}")

def batch_process_npy(input_dir, output_dir):
    """ 批量处理目录下的所有 .npy 文件 """
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".npy"):
            file_path = os.path.join(input_dir, file_name)
            process_and_save_slices(file_path, output_dir)

if __name__ == "__main__":
    input_dir = r"E:\脑龄\IXI\T1"  # 输入目录
    output_dir = r"E:\脑龄\IXI_2d\T1"  # 输出目录

    batch_process_npy(input_dir, output_dir)