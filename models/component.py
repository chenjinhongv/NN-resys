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
from torch.nn import Module, Linear, ReLU, Sequential, Embedding, ModuleList, ModuleDict, Parameter
from torch.nn.init import normal_
import hashlib


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
    def __init__(self, bucket_size, embedding_dim):
        super(HashEmbedding, self).__init__()
        self.embedding = Embedding(num_embeddings=bucket_size, embedding_dim=embedding_dim)
        self.bucket_size = bucket_size

    def forward(self, x):
        out = x.map_(x, self.hash_buckets)
        out = self.embedding(out)
        out = out.sum(dim=1)
        return out

    def hash_buckets(self, key, *karg):
        md5 = hashlib.md5()
        md5.update(str(key).encode('utf-8'))
        hash_key = md5.hexdigest()
        return int(hash_key, 16) % self.bucket_size


def hash_buckets(key, bucket_size):
    md5 = hashlib.md5()
    md5.update(str(key).encode('utf-8'))
    hash_key = md5.hexdigest()
    return int(hash_key, 16) % bucket_size


if __name__=="__main__":
    # print(hash_buckets(1246, bucket_size=10))
    model = FmInteraction(left_feature_num=4, right_feature_num=6, factor_num=10)
    left_ids = torch.tensor([0, 1, 3], dtype=torch.int64)
    left_vals = torch.tensor([[1, .5, 1]])
    right_ids = torch.tensor([2, 1, 5, 4], dtype=torch.int64)
    right_vals = torch.tensor([[1, 1, .6, .7]])
    out = model(left_ids, left_vals, right_ids, right_vals)
    print(out)
