# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : trainer.py
# Time       ：2022/5/23 15:51
# Author     ：huangtaogan
# Email      ：842960911@qq.com
# Description：
"""
from torch import optim
from tqdm import tqdm
import torch
import os
from tensorboardX import SummaryWriter


class Trainer:
    def __init__(self, train_data, val_data, model, max_epoch, device, loss_func, early_stop, optimizer, optim_hparas,
                 lr_decay=None, warm_up_steps=None, metrics=None, model_name=None, log_dir=None, save_path='./output'):
        """
        :param train_data: torch.utils.data.Dataloader Object
        :param val_data: torch.utils.data.Dataloader Object
        :param model: torch.nn.Module Object
        :param max_epoch: int
        :param device: str, would be use like model.to(device)
        :param loss_func:
        :param early_stop:
        :param optimizer:
        :param optim_hparas:
        :param lr_decay:
        :param warm_up_steps:
        :param metrics:
        :param model_name:
        :param log_dir:
        :param save_path:
        """

        self.train_data = train_data
        self.val_data = val_data
        self.max_epoch = max_epoch
        self.model = model
        self.optimizer = getattr(optim, optimizer)(self.model.parameters(), **optim_hparas)
        self.loss_func = loss_func
        self.early_stop = early_stop
        self.device = device
        self.metrics = metrics
        self.lr_decay = lr_decay
        self.base_lr = self.optimizer.defaults['lr']
        self.warm_up_steps = warm_up_steps

        if not model_name: model_name = type(model).__name__
        self.model_name = model_name
        # if not os.path.exists(log_dir): os.makedirs(log_dir)
        self.log_dir = log_dir
        self.summary_writer = SummaryWriter()
        if not os.path.exists(os.path.join(save_path, model_name)): os.makedirs(os.path.join(save_path, model_name))
        self.save_path = save_path

    def warming_up(self, global_step):
        self.optimizer.defaults['lr'] = self.base_lr * global_step/self.warm_up_steps

    def learning_rate_decay(self, global_step):
        if global_step % self.lr_decay['decay_steps'] == 0:
            self.optimizer.defaults['lr'] *= self.lr_decay['decay_rate']

    def train(self):
        min_loss = float('inf')
        un_promote_round = 0
        global_step = 0
        for epoch in range(self.max_epoch):
            self.model.train()
            self.model.to(self.device)
            for x, y in tqdm(iter(self.train_data)):
                self.optimizer.zero_grad()
                # x, y = x.to(self.device), y.to(self.device)
                for k in x:
                    x[k] = x[k].to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                loss = self.loss_func(pred, y)
                loss.backward()
                self.optimizer.step()
                global_step += 1
                loss = loss.detach().cpu().item()
                self.summary_writer.add_scalar(tag='train/loss', scalar_value=loss, global_step=global_step)
                if self.warm_up_steps and global_step < self.warm_up_steps:
                    self.warming_up(global_step=global_step)
                elif self.lr_decay:
                    self.learning_rate_decay(global_step=global_step)
            valid_loss = self.valid()
            self.summary_writer.add_scalar(tag='valid/loss', scalar_value=valid_loss, global_step=global_step)

            # save model
            is_save = valid_loss < min_loss
            if is_save:
                min_loss = valid_loss
                torch.save(self.model.state_dict(), os.path.join(self.save_path, self.model_name, 'final.pth'))
                un_promote_round = 0
            else:
                un_promote_round += 1

            # print
            print('epoch = {:4d}, valid_loss = {:.4f}, min_loss = {:.4f}, save_model = {}'.format(
                epoch, valid_loss, min_loss, is_save
            ))

            # early stop
            if un_promote_round > self.early_stop:
                break

        print("Finished training after {} epochs".format(epoch))

    def valid(self):
        self.model.eval()
        self.model.to(self.device)
        all_pred = []
        all_y = []
        for x, y in tqdm(iter(self.val_data)):
            for k in x:
                x[k] = x[k].to(self.device)
            y = y.to(self.device)
            pred = self.model(x)
            all_pred.append(pred)
            all_y.append(y)
        all_pred = torch.cat(all_pred, 0)
        all_y = torch.cat(all_y, 0)
        valid_loss = self.loss_func(all_pred, all_y)
        valid_loss = valid_loss.cpu().item()
        return valid_loss
