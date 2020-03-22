# -*- coding:utf-8 -*-

import os
import random
import math

import tqdm

import numpy as np
import torch
class GenDataIter(object):
    """ Toy data iter to load digits"""
    def __init__(self, data_file, batch_size):
        super(GenDataIter, self).__init__()
        self.batch_size = batch_size
        self.data_lis1, self.data_lis2 = self.read_file(data_file)
        self.data_num = len(self.data_lis1)
        self.indices = range(self.data_num)
        self.num_batches = int(math.ceil(float(self.data_num)/self.batch_size))
        self.idx = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self):
        self.idx = 0
        temp = list(zip(self.data_lis1, self.data_lis2)) 
        random.shuffle(temp) 
        self.data_lis1, self.data_lis2 = zip(*temp) 

    def next(self):
        if self.idx + self.batch_size >= self.data_num:
            raise StopIteration
        index = self.indices[self.idx:self.idx+self.batch_size]
        d = [self.data_lis1[i] for i in index]
        d = torch.LongTensor(np.asarray(d, dtype='int64'))
        t = [self.data_lis2[i] for i in index]
        t = torch.LongTensor(np.asarray(t, dtype='int64'))
        data = torch.cat([torch.zeros(self.batch_size, 1).long(), d], dim=1)
        target = torch.cat([t, torch.zeros(self.batch_size, 1).long()], dim=1)
        self.idx += self.batch_size
        return data, target

    def read_file(self, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        lis1 = []
        lis2 = []
        for line in lines:
            l1 = (line.split('\t')[0]).strip().split(' ')
            l1 = [int(s) for s in l1]
            lis1.append(l1)
            l2 = (line.split('\t')[1]).strip().split(' ')
            l2 = [int(s) for s in l2]
            lis2.append(l2)
        return lis1, lis2

class DisDataIter(object):
    """ Toy data iter to load digits"""
    def __init__(self, real_data_file, fake_data_file, batch_size):
        super(DisDataIter, self).__init__()
        self.batch_size = batch_size
        real_data_lis = self.read_file(real_data_file)
        fake_data_lis = self.read_file(fake_data_file)
        self.data = real_data_lis + fake_data_lis
        self.labels = [1 for _ in range(len(real_data_lis))] +\
                        [0 for _ in range(len(fake_data_lis))]
        self.pairs = list(zip(self.data, self.labels))
        self.data_num = len(self.pairs)
        self.indices = range(self.data_num)
        self.num_batches = int(math.ceil(float(self.data_num)/self.batch_size))
        self.idx = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self):
        self.idx = 0
        random.shuffle(self.pairs)

    def next(self):
        if self.idx >= self.data_num:
            raise StopIteration
        index = self.indices[self.idx:self.idx+self.batch_size]
        pairs = [self.pairs[i] for i in index]
        data = [p[0] for p in pairs]
        label = [p[1] for p in pairs]
        data = torch.LongTensor(np.asarray(data, dtype='int64'))
        label = torch.LongTensor(np.asarray(label, dtype='int64'))
        self.idx += self.batch_size
        return data, label

    def read_file(self, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        lis = []
        for line in lines:
            l = (line.strip().split('\t')[1]).strip().split(' ')
            l = [int(s) for s in l]
            lis.append(l)
        return lis
