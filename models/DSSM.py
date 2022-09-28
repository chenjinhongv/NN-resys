# -*-coding:utf-8 -*-

"""
# File       : DSSM
# Time       ：2022/9/28 20:44
# Author     ：chenjinhongv
# Email      ：842960911@qq.com
"""

import torch
from torch.nn import Module
from .component import MlpLayer
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")


def _get_text_embeddings(texts):
    encoded_input = tokenizer(texts, return_tensors='pt', padding=True)
    output = model(**encoded_input)
    return output['last_hidden_state']


class DSSM(Module):
    def __init__(self):
        super(DSSM, self).__init__()
        self.title_mlp = MlpLayer(input_dim=768, mlp_structure=[128,64,32])
        self.query_mlp = MlpLayer(input_dim=768, mlp_structure=[128,64,32])

    def forward(self, title, query):
        title = title.mean(dim=1)
        title = self.title_mlp(title)
        query = query.mean(dim=1)
        query = self.query_mlp(query)
        out = torch.mul(title, query).mean(dim=1)
        return out