# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : example_data.py
# Time       ：2022/5/23 16:14
# Author     ：huangtaogan
# Email      ：842960911@qq.com
# Description：
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import warnings
if not __name__ == "__main__":
    from .utils import map_style_collate_fn

warnings.filterwarnings('ignore')

MOVIE_LENS_PATH = './cache/ml-1m/'


class MovieLensDataset(Dataset):
    def __init__(self, mode="debug", model='WideDeep'):
        super(MovieLensDataset, self).__init__()
        self.data = self.get_sample()
        # self.mode = mode
        self.model = model
        ages = list(self.data['age'].drop_duplicates())
        age_index = dict(zip(ages, range(len(ages))))
        self.data['age'] = self.data['age'].map(age_index)
        genres = ['Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary',
                    'Drama',
                    'Fantasy',
                    'Film-Noir',
                    'Horror',
                    'Musical',
                    'Mystery',
                    'Romance',
                    'Sci-Fi',
                    'Thriller',
                    'War',
                    'Western']
        genre_index = dict(zip(genres, range(1, len(genres)+1)))
        self.data['genres'] = self.data['genres'].map(lambda x: [genre_index[genre] for genre in x.split('|')])
        self.data['rating'] = self.data['rating'].map(lambda x: 1 if x >= 4 else 0)
        self.data['gender'] = self.data['gender'].map(lambda x: 1 if x == 'F' else 0)
        if mode == 'debug':
            self.data = self.data.head(50)
        elif mode == 'test':
            _, self.data = train_test_split(self.data, test_size=.2, random_state=33)
        elif mode == 'train':
            self.data, _ = train_test_split(self.data, test_size=.2, random_state=33)

    @staticmethod
    def get_sample():
        if os.path.exists(os.path.join(MOVIE_LENS_PATH, 'sample.csv')):
            data = pd.read_csv(os.path.join(MOVIE_LENS_PATH, 'sample.csv'))
        else:
            ratings = pd.read_csv(os.path.join(MOVIE_LENS_PATH, 'ratings.dat'), sep='::', header=None,
                                  names=['userId', 'movieId', 'rating', 'timestamp'])
            movies = pd.read_csv(os.path.join(MOVIE_LENS_PATH, 'movies.dat'), sep='::', header=None,
                                 names=['movieId', 'title', 'genres'], encoding='ISO-8859-1')
            users = pd.read_csv(os.path.join(MOVIE_LENS_PATH, 'users.dat'), sep='::', header=None,
                                names=['userId', 'gender', 'age', 'occupation', 'zip-code'])
            data = pd.merge(ratings, movies, how='left', on='movieId')
            data = pd.merge(data, users, how='left', on='userId')
            data.reset_index(drop=True)
            data['gender'] = data['gender'].map(lambda x: 1 if x == 'F' else 0)
            data.to_csv(os.path.join(MOVIE_LENS_PATH, 'sample.csv'), index=False)
        return data

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        y = torch.tensor([row['rating']], dtype=torch.float)
        x = dict()
        x['userId'] = torch.tensor([(row['userId'])])
        x['movieId'] = torch.tensor([(row['movieId'])])
        x['genres'] = torch.tensor(row['genres'])
        x['gender'] = torch.tensor([row['gender']])
        x['age'] = torch.tensor([row['age']])
        x['occupation'] = torch.tensor([row['occupation']])
        return x, y

    def __len__(self):
        return self.data.shape[0]


def get_movie_lens_dataloader(mode='debug', batch_size=25):
    ds = MovieLensDataset(mode=mode)
    is_shuffle = mode == 'train'
    dl = DataLoader(dataset=ds, batch_size=batch_size, collate_fn=map_style_collate_fn, shuffle=is_shuffle)
    return dl


if __name__ == "__main__":

    MOVIE_LENS_PATH = '../cache/ml-1m/'

    data = MovieLensDataset.get_sample()
    row = data.iloc[10]
    print(row['userId'])
    print(row)

    a = {'genres': torch.tensor([1, 2, ])}
    b = {'genres': torch.tensor([4, 1, 3])}
    d = [a, b]

    class Adata(Dataset):
        def __init__(self):
            super(Adata, self).__init__()
            self.data = d

        def __getitem__(self, idx):
            return self.data[idx]

        def __len__(self):
            return len(self.data)


    def map_style_collate_fn(batch):
        collated_batch = dict()
        elem = batch[0]
        if isinstance(elem, dict):
            for key in elem:
                collated_batch[key] = pad_sequence([ele[key] for ele in batch], batch_first=True)
        else:
            raise RuntimeError('each element in list of batch should be a dict')
        return collated_batch

    adataset = Adata()
    ds = DataLoader(dataset=adataset, batch_size=2, collate_fn=map_style_collate_fn)
    for i in ds:
        print(i)
