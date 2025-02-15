import torch
import models
from options import args
import utility
import models
import Data
from torch.utils.data import DataLoader


if __name__ == '__main__':
    torch.manual_seed(args.seed)
    modal1_loader = Data.Brain_image(args.modal1, args)
    modal2_loader = Data.Brain_image(args.modal2, args)
    print("modal1");
    for images, labels, names in modal1_loader:
        # 在这里处理每个批次的数据
        print(images.shape, labels.shape, names)
    print("\nmodal2");
    for images, labels, names in modal2_loader:
        # 在这里处理每个批次的数据
        print(images.shape, labels.shape, names)
    print("\nmodal1_length", len(modal1_loader))
    print("modal2_length", len(modal2_loader))
    checkpoint = utility.checkpoint(args)
    if checkpoint.ok:


        model = models.ConvNet(args)
        loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, model, loss, checkpoint)
        while not t.terminate():
            t.train()
            t.test()

        checkpoint.done()


