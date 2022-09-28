# -*-coding:utf-8 -*-

"""
# File       : ESMM
# Time       ：2022/9/25 21:41
# Author     ：chenjinhongv
# Email      ：842960911@qq.com
"""

from torch.nn import Embedding, Module
from .component import MlpLayer, FieldEmbedding
from .utils import get_embedding_dict


class ESMM(Module):
    def __init__(self, user_field_features, item_field_features, context_field_features):
        super(ESMM, self).__init__()
        self.user_feature_embeddings = get_embedding_dict(user_field_features)
        self.item_feature_embeddings = get_embedding_dict(item_field_features)
        self.context_feature_embeddings = get_embedding_dict(context_field_features)

    def forward(self, batch):
        for feature in self.user_feature_embeddings:
            pass