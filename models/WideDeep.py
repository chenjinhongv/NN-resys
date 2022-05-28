# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : WideDeep.py
# Time       ：2022/5/19 16:45
# Author     ：huangtaogan
# Email      ：842960911@qq.com
# Description：
"""

import torch
from torch.nn import Module, Linear, Embedding, Sigmoid, ModuleDict
from .component import MlpLayer, FieldEmbedding


class WideDeep(Module):
    def __init__(self, deep_features, wide_features, deep_features_conf, wide_features_conf, deep_structure=[128, 64, 32]):
        """
        :param deep_features:features(list[str]) which will be used as input of wide&deep model's deep part
        :param wide_features:features(list[str]) which will be used as input of wide&deep model's wide part
        :param deep_features_conf:features conf (dict) keyed by features name (str),configs for every feature below:
            type: feature type(str) with options(continuous, categorical, multi_categorical)
            num_categories: int, should be given when type is categorical or multi_categorical
            embedding_dim: int, should be given when type is categorical or multi_categorical
        :param wide_features_conf: same as above
        :param deep_structure:node nums(list[int]) for mlp
        """
        super(WideDeep, self).__init__()

        self.deep_features = deep_features
        self.wide_features = wide_features
        self.deep_features_conf = deep_features_conf
        self.wide_features_conf = wide_features_conf
        # self.features_conf = features_conf

        # input embeddings
        self.wide_embeddings = self._get_wide_embeddings()
        self.deep_embeddings = self._get_deep_embeddings()

        deep_input_dim = self._get_deep_input_dim()
        wide_input_dim = self._get_wide_input_dim()
        self.deep = MlpLayer(input_dim=deep_input_dim, mlp_structure=deep_structure)

        self.out_layer = Linear(in_features=wide_input_dim + deep_structure[-1], out_features=1)
        self.out_activation = Sigmoid()

    def forward(self, batch):
        wide_inputs = []
        for feature in self.wide_features:
            try:
                emb = self.wide_embeddings[feature]
                wide_inputs.append(emb(batch[feature]))
            except KeyError as e:
                wide_inputs.append(batch[feature])
            # emb = self.wide_embeddings.get(feature, [])
            # if emb:
            #     wide_inputs.append(emb(batch[feature]))
            # else:
            #     wide_inputs.append(batch[feature])
        wide = torch.cat(wide_inputs, dim=1)

        deep_inputs = []
        for feature in self.deep_features:
            try:
                emb = self.deep_embeddings[feature]
                deep_inputs.append(emb(batch[feature]))
            except KeyError as e:
                deep_inputs.append(batch[feature])
            # emb = self.deep_embeddings.get(feature, [])
            # if emb:
            #     deep_inputs.append(emb(batch[feature]))
            # else:
            #     deep_inputs.append(batch[feature])
        deep = torch.cat(deep_inputs, dim=1)

        deep = self.deep(deep)
        out = torch.cat([wide, deep], dim=1)
        out = self.out_layer(out)
        out = self.out_activation(out)
        return out

    def _get_deep_input_dim(self):
        deep_input_dim = 0
        for feature in self.deep_features:
            conf = self.deep_features_conf[feature]
            if conf['type'] in ['categorical', 'multi_categorical']:
                deep_input_dim += conf['embedding_dim']
            else:
                deep_input_dim += 1
        return deep_input_dim

    def _get_wide_input_dim(self):
        wide_input_dim = 0
        for feature in self.wide_features:
            conf = self.wide_features_conf[feature]
            if conf['type'] in ['categorical', 'multi_categorical']:
                wide_input_dim += conf['embedding_dim']
            else:
                wide_input_dim += 1
        return wide_input_dim

    def _get_wide_embeddings(self):
        wide_embeddings = ModuleDict()
        for feature in self.wide_features:
            feature_conf = self.wide_features_conf.get(feature, [])
            if feature_conf:
                if feature_conf['type'] == 'categorical':
                    wide_embeddings[feature] = FieldEmbedding(num_embeddings=feature_conf['num_categories'], embedding_dim=feature_conf['embedding_dim'])
                elif feature_conf['type'] == 'multi_categorical':
                    wide_embeddings[feature] = FieldEmbedding(num_embeddings=feature_conf['num_categories']+1, embedding_dim=feature_conf['embedding_dim'])
        return wide_embeddings

    def _get_deep_embeddings(self):
        deep_embeddings = ModuleDict()
        for feature in self.deep_features:
            feature_conf = self.deep_features_conf.get(feature, [])
            if feature_conf:
                if feature_conf['type'] == 'categorical':
                    deep_embeddings[feature] = FieldEmbedding(num_embeddings=feature_conf['num_categories'], embedding_dim=feature_conf['embedding_dim'])
                elif feature_conf['type'] == 'multi_categorical':
                    deep_embeddings[feature] = FieldEmbedding(num_embeddings=feature_conf['num_categories']+1, embedding_dim=feature_conf['embedding_dim'])
        return deep_embeddings


if __name__ == "__main__":
    model = WideDeep(wide_input_dim=10, deep_input_dim=10)
