# -*-coding:utf-8 -*-

"""
# File       : demo
# Time       ：2022/9/27 13:54
# Author     ：chenjinhongv
# Email      ：842960911@qq.com
"""

from models.DSSM import DSSM
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")


def _get_text_embeddings(texts):
    encoded_input = tokenizer(texts, return_tensors='pt', padding=True)
    output = model(**encoded_input)
    return output['last_hidden_state']


if __name__ == '__main__':
    dssm = DSSM()
    title = ["Replace me by any text you'd like.", 'her name plz']
    query = ["Replace me by any text you'd like.", 'her name plz']
    title = _get_text_embeddings(title)
    query = _get_text_embeddings(query)
    labels = [1, 0]
    preds = dssm(title, query)
    print(preds)
