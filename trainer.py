import os
import math
import matplotlib
matplotlib.use('Agg')
import utility
import torch
import numpy as np
from decimal import Decimal
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

class Trainer():
    def __init__(self, args, loader_train, loader_test, my_model, my_loss, ckp):
        self.args = args
        self.ckp = ckp
        self.loader_train = loader_train
        self.loader_test = loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8


    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1

        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()

        if epoch == 1 and self.args.load == '.':
            # 如果是第一轮训练且未加载预训练模型
            self.loader_train.dataset.first_epoch = True
            # adjust learning rate
            lr = 1e-3
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # 对于其他轮次的训练
            self.loader_train.dataset.first_epoch = False
            # adjust learning rate
            lr = self.args.lr * (2 ** -(epoch // 30))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))

        for batch, (img1, img2, true_age, name) in enumerate(self.loader_train):

            timer_data.hold()
            self.optimizer.zero_grad()

            # inference
            self.model.get_model()

            pred_age = self.model(img1, img2)

            # loss function
            loss = self.loss(pred_age, true_age)

            # backward
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

        target = self.model
        torch.save(
            target.state_dict(),
            os.path.join(self.ckp.dir, 'model', 'model_latest.pt')
        )
        if epoch % self.args.save_every == 0:
            torch.save(
                target.state_dict(),
                os.path.join(self.ckp.dir, 'model', 'model_{}.pt'.format(epoch))
            )
            self.ckp.write_log('save ckpt epoch{:.4f}'.format(epoch))

    def test(self):
        self.model.eval()

        with torch.no_grad():
            if self.args.test_only:
                logger = print
            else:
                logger = self.ckp.write_log

            pred_age_list = []
            true_age_list = []

            for idx_img, (img1, img2, true_age, name) in tqdm(enumerate(self.loader_test), total=len(self.loader_test)):
                pred_age = self.model(img1, img2)
                pred_age_list.append(pred_age)
                true_age_list.append(true_age)

            # 将预测年龄和真实年龄列表转换为张量
            pred_ages = torch.cat(pred_age_list, dim=0)
            true_ages = torch.cat(true_age_list, dim=0)

            # 计算 MSE 和 MAE
            mse = F.mse_loss(pred_ages, true_ages).item()
            mae = F.l1_loss(pred_ages, true_ages).item()

            logger('[{}]\tMSE: {:.4f} MAE: {:.4f}'.format(
                self.args.data_test,
                mse,
                mae
            ))

        if not self.args.test_only: #training mode and save the best model
            if self.mse_max is None or self.mse_max < mse:
                self.mse_max = mse
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.ckp.dir, 'model', 'model_best.pt')
                )
                logger('save ckpt MSE:{:.4f}'.format(mse))


    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
