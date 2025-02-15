import os
from importlib import import_module
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, args, ckp): # ckp: checkpoint
        super(Model, self).__init__()
        print('Making model...')

        self.n_GPUs = args.n_GPUs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_models = args.save_models

        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)

        if not args.cpu and args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))

        self.load(
            ckp.dir,
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
        )

    def forward(self, x):
        return self.model(x)

    def get_model(self):
        if self.n_GPUs <= 1 or self.cpu:
            return self.model
        else:
            return self.model.module

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def save(self, apath, epoch, is_best=False):
        target = self.get_model()
        torch.save(
            target.state_dict(),
            os.path.join(apath, 'model', 'model_latest.pt')
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', 'model_best.pt')
            )

        if self.save_models:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', 'model_{}.pt'.format(epoch))
            )

    def load(self, apath, pre_train='.', resume=-1, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        if resume == 0:
            if pre_train != 'None':
                pretrained_dict = torch.load(pre_train)
                if 'model' in pretrained_dict.keys():
                    pretrained_dict = pretrained_dict['model']['sd']

                model_dict = self.get_model().state_dict()

                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.get_model().load_state_dict(model_dict)
                print('load from x4 pre-trained model')

        elif resume > 0:
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_{}.pt'.format(resume)),
                    **kwargs
                ),
                strict=False
            )
            print('load from model_' + str(resume) + '.pt')
