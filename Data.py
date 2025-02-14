import torch
from torch.utils.data import Dataset as image_dataset
from torch_geometric.data import Dataset as graph_dataset
from torch_geometric.data import Data
import os
import numpy as np


class Brain_image(image_dataset):
    def __init__(self, modality, args):
        self.data_dir = args.data_dir
        self.modality = modality
        self.image_list, self.label_list, self.name_list = self.get_data()

    def __getitem__(self, index):
        # get item by index
        image, label, name = np.load(self.image_list[index]), np.load(self.label_list[index]), self.name_list[index]

        # transform numpy to tensor
        image = torch.from_numpy(image)

        # add channel dimension for image
        image = torch.unsqueeze(image, dim=0)

        return image, label, name

    def __len__(self):
        return len(self.image_list)

    def get_data(self):
        image_list = list()
        label_list = list()
        name_list = list()

        # define file paths
        image_path = os.path.join(self.data_dir, str(self.modality))
        label_path = os.path.join(self.data_dir, 'Age')

        sub_dir = os.listdir(image_path)
        sub_dir.sort(key=lambda x: int(x[:-4]))

        # load data and label
        for name in sub_dir:
            image = os.path.join(image_path, name)
            label = os.path.join(label_path, name)
            image_list.append(image)
            label_list.append(label)
            name_list.append(name)

        return image_list, label_list, name_list