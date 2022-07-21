# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : hash.py
# Time       ：2022/7/21 16:31
# Author     ：huangtaogan
# Email      ：842960911@qq.com
# Description：
"""
from typing import List


class Hash:
    def __init__(self, a, b, mod):
        self.a = a
        self.b = b
        self.mod = mod

    def __call__(self, key):
        return (self.a * key + self.b) % self.mod


class HashFactory:
    def __init__(self, a_list: List[int], b_list: List[int], mod: int):
        """
        desc: create series of hash functions which accept int key and return hash val = (a_list[i] * key + b_list[i]) % mod
        :param a_list:
        :param b_list:
        :param mod: max Prime number which is smaller than the excepted max hash val
        """
        self.a_list = a_list
        self.b_list = b_list
        self.mod = mod

    def get_hash_funcs(self):
        hash_funcs = []
        for a, b in zip(self.a_list, self.b_list):
            hash_funcs.append(Hash(a=a, b=b, mod=self.mod))
        return hash_funcs


if __name__ == "__main__":
    import random
    a_list = [4, 9, 6, 7]
    b_list = [8, 10, 47, 45]
    mod = 31
    hash_funcs = HashFactory(a_list=a_list, b_list=b_list, mod=mod).get_hash_funcs()
    keys = [random.randint(10, 10000) for _ in range(10)]
    for hash_func in hash_funcs:
        print(hash_func)
        print([hash_func(key) for key in keys])
