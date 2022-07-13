# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : component.py
# Time       ：2022/5/23 14:42
# Author     ：huangtaogan
# Email      ：842960911@qq.com
# Description：
"""

from torch.nn import Module, Linear, ReLU, Sequential, Embedding, ModuleList, ModuleDict
import hashlib


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
    print(hash_buckets(1246, bucket_size=10))
