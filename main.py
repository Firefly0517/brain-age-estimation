import torch
from options import args
import utility
import dataloader
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import models
import loss
from trainer import Trainer

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    print("cuda_available:", torch.cuda.is_available())
    checkpoint = utility.checkpoint(args)
    if checkpoint.ok:
        full_dataset = dataloader.Brain_image(args)

        # print("modal");
        # for images1, images2, labels, names in full_dataset:
        #     print(images1.shape, images2.shape,labels.shape, names)
        #     break
        # torch.Size([1, 160, 192, 160]) channel=1

        # 划分数据集
        dataset_size = len(full_dataset)
        train_ratio = args.train_ratio  # 训练集比例
        val_ratio = args.val_ratio  # 验证集比例
        test_ratio = args.test_ratio  # 测试集比例

        train_size = int(train_ratio * dataset_size)
        val_size = int(val_ratio * dataset_size)
        test_size = dataset_size - train_size - val_size
        print("train_size:",train_size, "val_size:",val_size)

        indices = list(range(dataset_size))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        # create new dateset
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        test_dataset = Subset(full_dataset, test_indices)

        # create new dataloader
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=not args.cpu)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=not args.cpu)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=not args.cpu)

        # 打印训练集数据信息
        # print("Training data:")
        # for images1, images2, labels, names in train_loader:
        #     print(images1.shape, images2.shape, labels.shape, names)
        #     break
        # torch.Size([2, 1, 160, 192, 160]) batch=2 channel=1


        model = models.Model(args, checkpoint)

        loss = loss.Loss(args, checkpoint) if not args.test_only else None
        if args.test_only:
            t = Trainer(args, train_loader, test_loader, model, loss, checkpoint)
        else:
            t = Trainer(args, train_loader, val_loader, model, loss, checkpoint)
        while not t.terminate():
            t.train()
            t.test()
        checkpoint.done()


