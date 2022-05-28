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
