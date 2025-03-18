import os


def rename_files_in_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            old_name = os.path.join(root, file)
            print(old_name)
            new_name = file[37:58] + '.npy'
            print(new_name)
            new_name = os.path.join(root, new_name)
            os.rename(old_name, new_name)


if __name__ == '__main__':
    folder_path = r'/data2/wangchangmiao/ADNI/ADNI_T1/ADNI_T1_558_norm'
    rename_files_in_folder(folder_path)