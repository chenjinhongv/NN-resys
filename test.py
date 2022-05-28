# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : test.py
# Time       ：2022/5/23 14:43
# Author     ：huangtaogan
# Email      ：842960911@qq.com
# Description：
"""

from models.WideDeep import WideDeep
from data.example_data import get_movie_lens_dataloader
from utils.trainer import Trainer
from torch.nn import BCELoss

# movie_data = get_movie_lens_dataloader()
# movie_data_iter = iter(movie_data)
# x, y = movie_data_iter.next()
# print(x.keys())

# load movie lens data
movie_data_train = get_movie_lens_dataloader(mode='train')
movie_data_valid = get_movie_lens_dataloader(mode='test')

# model init
deep_features = ['gender', 'genres', 'age', 'occupation']
deep_features_conf = {
    'gender': {'type': 'continuous'},
    'genres': {'type': 'multi_categorical', 'num_categories': 18, 'embedding_dim': 4},
    'age': {'type': 'categorical', 'num_categories': 7, 'embedding_dim': 4},
    'occupation': {'type': 'categorical', 'num_categories': 21, 'embedding_dim': 4}
}
wide_features = ['gender', 'genres', 'age', 'occupation']
wide_features_conf = {
    'gender': {'type': 'continuous'},
    'genres': {'type': 'multi_categorical', 'num_categories': 18, 'embedding_dim': 1},
    'age': {'type': 'categorical', 'num_categories': 7, 'embedding_dim': 1},
    'occupation': {'type': 'categorical', 'num_categories': 21, 'embedding_dim': 1}
}
model = WideDeep(deep_features=wide_features, deep_features_conf=deep_features_conf, wide_features=wide_features,
                 wide_features_conf=wide_features_conf)

loss_func = BCELoss()

trainer = Trainer(train_data=movie_data_train, val_data=movie_data_valid, model=model, max_epoch=100, device='cuda',
                  loss_func=loss_func, early_stop=10, optimizer='SGD', optim_hparas={'lr': 0.0001})

trainer.train()
