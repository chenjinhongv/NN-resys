# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : utils.py
# Time       ：2022/5/24 16:06
# Author     ：huangtaogan
# Email      ：842960911@qq.com
# Description：
"""

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class Process:
    """
    process origin dict data with structure{'feature_name': data}) into structure {'feature_name': tensor}
    """
    def __init__(self):
        pass


def map_style_collate_fn(batch):
    elem = batch[0]
    if isinstance(elem, dict):
        return {key: pad_sequence([ele[key] for ele in batch], batch_first=True) for key in elem}
    elif isinstance(elem, tuple):
        return tuple(map_style_collate_fn([ele[i] for ele in batch]) for i in range(len(elem)))
    elif isinstance(elem, torch.Tensor):
        return pad_sequence(batch, batch_first=True)
    else:
        raise RuntimeError('each element in list of batch should be torch.Tenser or dict with value of torch.Tensor')
