import os


def rename_files_in_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            old_name = os.path.join(root, file)
            parts = file.split('IXI')
            if len(parts) > 1:
                new_name = parts[1].split('-')[0] + '.npy'
                new_name = os.path.join(root, new_name)
                os.rename(old_name, new_name)


if __name__ == '__main__':
    folder_path = '../data/T1'
    rename_files_in_folder(folder_path)