import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')  # 强制使用 TkAgg 后端



def load_npy(file_path):
    # 加载.npy文件
    return np.load(file_path)


def slice_mri_data(data):
    # 获取数据的形状
    x, y, z = data.shape

    # 在三个方向中间切片
    axial_slice = data[x // 2, :, :]  # 轴向切片 (沿着X轴切片)
    coronal_slice = data[:, y // 2, :]  # 冠状面切片 (沿着Y轴切片)
    sagittal_slice = data[:, :, z // 2]  # 矢状面切片 (沿着Z轴切片)

    return axial_slice, coronal_slice, sagittal_slice


def display_slices(axial, coronal, sagittal):
    # 使用Matplotlib展示三个方向的切片
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

    plt.show()


def main(file_path):
    # 加载MRI数据
    mri_data = load_npy(file_path)

    # 获取三个方向的切片
    axial_slice, coronal_slice, sagittal_slice = slice_mri_data(mri_data)

    # 展示切片
    display_slices(axial_slice, coronal_slice, sagittal_slice)


if __name__ == "__main__":
    file_path = r'E:\脑龄\data\DTI\002.npy'  # 将此处路径替换为实际的npy文件路径
    main(file_path)
