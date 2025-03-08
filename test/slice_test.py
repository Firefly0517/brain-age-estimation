import numpy as np
import os
import matplotlib.pyplot as plt

def load_npy(file_path):
    """ 加载 .npy 文件 """
    return np.load(file_path)

def slice_mri_data(data):
    """ 获取 MRI 数据的三个方向切片 """
    x, y, z = data.shape
    axial_slice = data[x // 2, :, :]  # 轴向 (X)
    coronal_slice = data[:, y // 2, :]  # 冠状 (Y)
    sagittal_slice = data[:, :, z // 2]  # 矢状 (Z)
    return axial_slice, coronal_slice, sagittal_slice

def save_slice(slice_data, save_path):
    """ 保存切片为 .npy 文件 """
    np.save(save_path, slice_data)

def display_slices(axial, coronal, sagittal, save_path):
    """ 保存切片的展示图像 """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(axial, cmap='gray')
    axes[0].set_title("Axial Slice")
    axes[0].axis('off')

    axes[1].imshow(coronal, cmap='gray')
    axes[1].set_title("Coronal Slice")
    axes[1].axis('off')

    axes[2].imshow(sagittal, cmap='gray')
    axes[2].set_title("Sagittal Slice")
    axes[2].axis('off')

    # 保存图片而不是显示
    plt.savefig(save_path, bbox_inches='tight')  # 保存为图像文件
    plt.close(fig)  # 关闭图形以释放资源

def process_and_save_slices(file_path, output_dir):
    """ 处理单个 .npy 文件并保存切片 """
    # 读取数据
    mri_data = load_npy(file_path)

    # 获取切片
    axial_slice, coronal_slice, sagittal_slice = slice_mri_data(mri_data)

    # 获取原文件名（不带后缀）
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 定义保存路径
    save_paths = [
        os.path.join(output_dir, f"{file_name}_1.png"),  # 轴向
        os.path.join(output_dir, f"{file_name}_2.png"),  # 冠状
        os.path.join(output_dir, f"{file_name}_3.png")  # 矢状
    ]

    # 依次保存
    for slice_data, save_path in zip([axial_slice, coronal_slice, sagittal_slice], save_paths):
        display_slices(slice_data, slice_data, slice_data, save_path)  # 传入保存路径

    print(f"Processed: {file_path}")

def batch_process_npy(input_dir, output_dir):
    """ 批量处理目录下的所有 .npy 文件 """
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".npy"):  # 只处理 .npy 文件
            file_path = os.path.join(input_dir, file_name)
            process_and_save_slices(file_path, output_dir)

if __name__ == "__main__":
    input_dir = r"E:\脑龄\IXI\DTI"  # 你的 .npy 文件目录
    output_dir = r"E:\脑龄\IXI_2d\DTI"  # 切片保存目录

    batch_process_npy(input_dir, output_dir)
