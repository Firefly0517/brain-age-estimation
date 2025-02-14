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

    # checkpoint = utility.checkpoint(args)
    # if checkpoint.ok:
    #                     ## data loader
    #     # modal2_loader = dataloader.Brain_image(args.modal2, args)
    #     model = models.ConvNet(args)
    #     loss = loss.Loss(args, checkpoint) if not args.test_only else None
    #     t = Trainer(args, loader, model, loss, checkpoint)
    #     while not t.terminate():
    #         t.train()
    #         t.test()
    #
    #     checkpoint.done()


