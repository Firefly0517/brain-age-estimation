import torch
from torch.utils.data import Dataset as image_dataset
import os
import numpy as np


class Brain_image(image_dataset):
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.modal1 = args.modal1
        self.modal2 = args.modal2
        self.image1_list, self.image2_list, self.label_list, self.gender_list, self.name_list = self.get_data()

    def __getitem__(self, index):
        # get item by index
        image1, image2, age, gender, name = (np.load(self.image1_list[index]), np.load(self.image2_list[index]),
                                       np.load(self.label_list[index]), np.load(self.gender_list[index]),self.name_list[index])

        # transform numpy to tensor
        image1 = torch.from_numpy(image1)
        image2 = torch.from_numpy(image2)

        # add channel dimension for image
        image1 = torch.unsqueeze(image1, dim=0)
        image2 = torch.unsqueeze(image2, dim=0)

        return image1, image2, age, gender, name

    def __len__(self):
        return len(self.image1_list)

    def get_data(self):
        image1_list = list()
        image2_list = list()
        age_list = list()
        gender_list = list()
        name_list = list()

        # define file paths
        image1_path = os.path.join(self.data_dir, str(self.modal1))
        image2_path = os.path.join(self.data_dir, str(self.modal2))
        age_path = os.path.join(self.data_dir, 'Age')
        gender_path = os.path.join(self.data_dir, 'Gender')

        sub_dir = os.listdir(image1_path)
        # sub_dir.sort(key=lambda x: int(x[:-4])) # 排序

        # load data and label
        for name in sub_dir:
            image1 = os.path.join(image1_path, name)
            image2 = os.path.join(image2_path, name)
            age = os.path.join(age_path, name)
            gender = os.path.join(gender_path, name)

            if not all([os.path.exists(image2), os.path.exists(age), os.path.exists(gender)]):
                print("name not all found!!!!")
                continue

            image1_list.append(image1)
            image2_list.append(image2)
            age_list.append(age)
            gender_list.append(gender)
            name_list.append(name)

        return image1_list, image2_list, age_list, gender_list, name_list
