# -*-coding:utf-8 -*-

"""
# File       : utils
# Time       ：2022/9/27 23:45
# Author     ：chenjinhongv
# Email      ：842960911@qq.com
"""
from torch.nn import ModuleDict
from .component import FieldEmbedding


def get_embedding_dict(features_conf):
    emb_dict = ModuleDict()
    for feature, conf in features_conf:
        emb_dict[feature] = FieldEmbedding(num_embeddings=conf['num_categories'], embedding_dim=conf['embedding_dim'],
                                           summation=conf['summation'])
    return emb_dict
