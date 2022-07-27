# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : component.py
# Time       ：2022/5/23 14:42
# Author     ：huangtaogan
# Email      ：842960911@qq.com
# Description：
"""
import torch
from torch import Tensor
from torch.nn import Module, Linear, ReLU, Sequential, Embedding, ModuleList, ModuleDict, Parameter, Sigmoid
from torch.nn.init import normal_

import numpy as np
import hashlib
from typing import List, Union, Dict
from utils.hash import HashFactory


class FmInteraction(Module):
    def __init__(self, left_feature_num, right_feature_num, factor_num):
        """
        the module process Factor Machine interaction between two fields features which called left features(field)
        and right features(field) below
        :param left_feature_num: left field's feature number. a feature's feature num = category number for category
        faeture and 1 for continuous feature
        :param right_feature_num: same as left_feature_num
        :param factor_num:
        """
        super(FmInteraction, self).__init__()
        self.left_feature_factors = Parameter(torch.empty(left_feature_num, factor_num))
        normal_(self.left_feature_factors)
        self.right_feature_factors = Parameter(torch.empty(right_feature_num, factor_num))
        normal_(self.right_feature_factors)

    def forward(self, left_ids, left_vals, right_ids, right_vals):
        """
        :param left_ids: non-zero feature's indexes of left field, tensor with shape of (non-zero feature num,) and dtype of interger
        :param left_vals: non-zero feature's values of left field, tensor with shape of (1, non-zero feature num)
        :param right_ids: same as left_ids
        :param right_vals: same as left_vals
        :return:
        """
        left_factors = self.left_feature_factors[left_ids, :]
        right_factors = self.right_feature_factors[right_ids, :]
        out = torch.mm(left_factors, right_factors.T) * torch.mm(left_vals.T, right_vals)
        out = out.sum()
        return out


class MlpLayer(Module):
    def __init__(self, input_dim, mlp_structure):
        super(MlpLayer, self).__init__()

        seq = ModuleList()
        for in_features, out_features in zip([input_dim] + mlp_structure[:-1], mlp_structure):
            seq.append(Linear(in_features, out_features))
            seq.append(ReLU())

        self.net = Sequential(*seq)

    def forward(self, x):
        return self.net(x)


class FieldEmbedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(FieldEmbedding, self).__init__()
        self.embedding = Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

    def forward(self, x):
        out = self.embedding(x)
        out = out.sum(dim=1)
        return out


class HashEmbedding(Module):
    def __init__(self, bucket_size: int, embedding_dim: int, hash_func: object = None):
        super(HashEmbedding, self).__init__()
        self.embedding = Embedding(num_embeddings=bucket_size, embedding_dim=embedding_dim)
        self.bucket_size = bucket_size
        self.hash_func = hash_func if hash_func else self.hash_buckets

    def forward(self, x: Tensor) -> Tensor:
        out = x.map_(x, self.hash_func)
        out = self.embedding(out)
        out = out.sum(dim=1)
        return out

    def set_hash_func(self, hash_func: object):
        self.hash_func = hash_func

    def hash_buckets(self, key, *karg):
        """
        desc: default hash func
        :param key:
        :return:
        """
        md5 = hashlib.md5()
        md5.update(str(key).encode('utf-8'))
        hash_key = md5.hexdigest()
        return int(hash_key, 16) % self.bucket_size


class MultiHashEmbedding(Module):
    def __init__(self, bucket_size: int, embedding_dim: int, embedding_count: int):
        super(MultiHashEmbedding, self).__init__()

        self.bucket_size = bucket_size
        self.a_list = Parameter(torch.tensor(np.random.choice(range(embedding_count*2), size=embedding_count, replace=False),
                                             dtype=torch.float64))
        self.b_list = Parameter(torch.tensor(np.random.choice(range(embedding_count*2), size=embedding_count, replace=False),
                                             dtype=torch.float64))
        # self.hash_funcs = [None] * embedding_count
        self.hash_embedding_list = ModuleList([HashEmbedding(bucket_size=bucket_size, embedding_dim=embedding_dim) for _ in range(embedding_count)])
        self._set_hash_funcs()

    def load_state_dict(self, state_dict: Union[Dict[str, Tensor], Dict[str, Tensor]],
                        strict: bool = True):
        super(MultiHashEmbedding, self).load_state_dict(state_dict=state_dict, strict=strict)
        self._set_hash_funcs()

    def _set_hash_funcs(self):
        hash_funcs = HashFactory(a_list=self.a_list, b_list=self.b_list, mod=self.bucket_size).get_hash_funcs()
        for hash_embedding, hash_func in zip(self.hash_embedding_list, hash_funcs):
            hash_embedding.set_hash_func(hash_func)

    def forward(self, x: Tensor) -> List[Tensor]:
        out = []
        for hash_embedding in self.hash_embedding_list:
            out.append(hash_embedding(x))
        return out


class SelfWeightedSummation(Module):
    def __init__(self, input_dim=768, mlp_structure=[128, 64, 32]):
        super(SelfWeightedSummation, self).__init__()
        self.mlp_layer = MlpLayer(input_dim=input_dim, mlp_structure=mlp_structure)
        self.linear = Linear(mlp_structure[-1], 1)
        self.sigmoid = Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: Tensor with shape of (batchsize, seq_len, embedding_dim)
        :return: out: Tensor with shape of (batchsize, embedding_dim)
        """
        weights = []
        for col in range(x.shape[1]):
            weight = self.mlp_layer(x[:, col, :])
            weight = self.linear(weight)
            weight = self.sigmoid(weight)
            weights.append(weight)
        weights = torch.concat(weights, dim=1)
        weights = weights.view((x.shape[0], 1, -1))
        out = torch.bmm(weights, x)
        out = out.view((x.shape[0], -1))
        return out


if __name__ == "__main__":
    # print(hash_buckets(1246, bucket_size=10))
    model = FmInteraction(left_feature_num=4, right_feature_num=6, factor_num=10)
    left_ids = torch.tensor([0, 1, 3], dtype=torch.int64)
    left_vals = torch.tensor([[1, .5, 1]])
    right_ids = torch.tensor([2, 1, 5, 4], dtype=torch.int64)
    right_vals = torch.tensor([[1, 1, .6, .7]])
    out = model(left_ids, left_vals, right_ids, right_vals)
    print(out)

    model = MultiHashEmbedding(bucket_size=10, embedding_dim=14, embedding_count=3)
    x1 = torch.randint(high=100, size=(3, 1))
    x2 = torch.randint(high=100, size=(3, 4))
    out1 = model(x1)
    out2 = model(x2)
    print(out1)
    print(out2)

    model = SelfWeightedSummation()
    x = torch.empty((12, 25, 768))
    pred = model(x)
    print(pred.shape)
    print(pred)
