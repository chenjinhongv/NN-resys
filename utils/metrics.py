# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : metrics.py
# Time       ：2022/5/23 15:51
# Author     ：huangtaogan
# Email      ：842960911@qq.com
# Description：
"""


class Trainer:
    def __init__(self, model, dataloader, optm, metrics, epoch, early_stop_round, log_dir):
        self.model = model
        self.dataloader = dataloader
        self.optm = optm
        self.metrics = metrics
        self.epoch = epoch
        self.early_stop_round = early_stop_round
        self.log_dir = log_dir

    def format_print(self):
        pass

    def train(self):
        pass
