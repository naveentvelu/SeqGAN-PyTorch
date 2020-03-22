# -*- coding:utf-8 -*-

import os
import random
import math

import argparse
import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from generator import Generator
from discriminator import Discriminator
from target_lstm import TargetLSTM
from rollout import Rollout
from data_iter import GenDataIter, DisDataIter

import helpers
# ================== Parameter Definition =================

parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--cuda', action='store', default=None, type=int)
opt = parser.parse_args()
print(opt)

# Data File with Input, Response, Input Emotion, Response emotion, 5 persona sentences
TRAIN_DATA_FILE = 'persona.txt'
TEST_DATA_FILE = 'persona_test.txt'

print("Genrating the real data samples....")
POSITIVE_FILE = 'real.data'
TEST_FILE = 'test.data'

# Generate Vocab
vocab = helpers.generate_vocab(TRAIN_DATA_FILE, TEST_DATA_FILE)

# Positive Data 
helpers.generate_data(vocab, TRAIN_DATA_FILE, POSITIVE_FILE)
helpers.generate_data(vocab, TEST_DATA_FILE, TEST_FILE)

# Basic Training Paramters
SEED = 88
BATCH_SIZE = 64
TOTAL_BATCH = 200
GENERATED_NUM = 10000
NEGATIVE_FILE = 'gene.data'
EVAL_FILE = 'eval.data'
VOCAB_SIZE = vocab.n_words
PRE_EPOCH_NUM = 120

if opt.cuda is not None and opt.cuda >= 0:
    torch.cuda.set_device(opt.cuda)
    opt.cuda = True

# Genrator Parameters
g_emb_dim = 32
g_hidden_dim = 32
g_sequence_len = 20

# Discriminator Parameters
d_emb_dim = 64
d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]

d_dropout = 0.75
d_num_class = 2



def generate_samples(model, batch_size, generated_num, input_file, output_file):
    samples = []
    with open(input_file, 'r') as f:
        lines = f.readlines()
    inputs = [] 
    num_b = int(generated_num / batch_size)
    for i in range(num_b):
        selected_lines = random.sample(lines, batch_size)
        inp = []
        for line in selected_lines:
            l = (line.strip().split('\t')[0]).strip().split(' ')
            l = [int(s) for s in l]
            inp.append(l)
        inputs.extend(inp)

        inp = torch.LongTensor(np.asarray(inp, dtype='int64'))
        sample = model.conditional_sample(inp).cpu().data.numpy().tolist()
        samples.extend(sample)
    with open(output_file, 'w') as fout:
        for inp, sample in zip(inputs, samples):
            string1 = ' '.join([str(s) for s in inp])
            string2 = ' '.join([str(s) for s in sample])
            fout.write('%s\t%s\n' % (string1, string2))

def train_epoch(model, data_iter, criterion, optimizer):
    total_loss = 0.
    total_words = 0.
    for (data, target) in data_iter:#tqdm(
        #data_iter, mininterval=2, desc=' - Training', leave=False):
        data = Variable(data)
        target = Variable(target)
        if opt.cuda:
            data, target = data.cuda(), target.cuda()
        target = target.contiguous().view(-1)
        pred = model.forward(data)
        loss = criterion(pred, target)
        total_loss += loss.item()
        total_words += data.size(0) * data.size(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    data_iter.reset()
    return math.exp(total_loss / total_words)

def eval_epoch(model, data_iter, criterion):
    total_loss = 0.
    total_words = 0.
    with torch.no_grad():
        for (data, target) in data_iter:#tqdm(
            #data_iter, mininterval=2, desc=' - Training', leave=False):
            data = Variable(data)
            target = Variable(target)
            if opt.cuda:
                data, target = data.cuda(), target.cuda()
            target = target.contiguous().view(-1)
            pred = model.forward(data)
            loss = criterion(pred, target)
            total_loss += loss.item()
            total_words += data.size(0) * data.size(1)
        data_iter.reset()

    assert total_words > 0  # Otherwise NullpointerException
    return math.exp(total_loss / total_words)

class GANLoss(nn.Module):
    """Reward-Refined NLLLoss Function for adversial training of Gnerator"""
    def __init__(self):
        super(GANLoss, self).__init__()

    def forward(self, prob, target, reward):
        """
        Args:
            prob: (N, C), torch Variable
            target : (N, ), torch Variable
            reward : (N, ), torch Variable
        """
        N = target.size(0)
        C = prob.size(1)
        one_hot = torch.zeros((N, C))
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, target.data.view((-1,1)), 1)
        one_hot = one_hot.type(torch.ByteTensor)
        one_hot = Variable(one_hot)
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        loss = torch.masked_select(prob, one_hot)
        loss = loss * reward
        loss =  -torch.sum(loss)
        return loss


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # Define Networks
    generator = Generator(VOCAB_SIZE, g_emb_dim, g_hidden_dim, opt.cuda)
    discriminator = Discriminator(d_num_class, VOCAB_SIZE, d_emb_dim, d_filter_sizes, d_num_filters, d_dropout)

    if opt.cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    # Load data from file
    gen_data_iter = GenDataIter(POSITIVE_FILE, BATCH_SIZE)

    # Pretrain Generator using MLE
    gen_criterion = nn.NLLLoss(reduction='sum')
    gen_optimizer = optim.Adam(generator.parameters())
    if opt.cuda:
        gen_criterion = gen_criterion.cuda()
    print('Pretrain Generator with MLE ...')
    for epoch in range(PRE_EPOCH_NUM):
        loss = train_epoch(generator, gen_data_iter, gen_criterion, gen_optimizer)
        print('Epoch [%d] Loss: %f'% (epoch, loss))

    # Pretrain Discriminator
    dis_criterion = nn.NLLLoss(reduction='sum')
    dis_optimizer = optim.Adam(discriminator.parameters())
    if opt.cuda:
        dis_criterion = dis_criterion.cuda()
    print('Pretrain Discriminator ...')
    for epoch in range(5):
        generate_samples(generator, BATCH_SIZE, GENERATED_NUM, POSITIVE_FILE, NEGATIVE_FILE)
        dis_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, BATCH_SIZE)
        for se in range(3):
            loss = train_epoch(discriminator, dis_data_iter, dis_criterion, dis_optimizer)
            print('Epoch [%d].[%d], loss: %f' % (epoch, se, loss))
    # Adversarial Training
    rollout = Rollout(generator, 0.8)
    print('#####################################################')
    print('Start Adeversatial Training...\n')
    gen_gan_loss = GANLoss()
    gen_gan_optm = optim.Adam(generator.parameters())
    if opt.cuda:
        gen_gan_loss = gen_gan_loss.cuda()
    gen_criterion = nn.NLLLoss(reduction='sum')
    if opt.cuda:
        gen_criterion = gen_criterion.cuda()
    dis_criterion = nn.NLLLoss(reduction='sum')
    dis_optimizer = optim.Adam(discriminator.parameters())
    if opt.cuda:
        dis_criterion = dis_criterion.cuda()
    for total_batch in range(TOTAL_BATCH):
        ## Train the generator for one step
        for it in range(1):
            with open(POSITIVE_FILE, 'r') as f:
                lines = f.readlines()
            selected_lines = random.sample(lines, BATCH_SIZE)
            inp = []
            for line in selected_lines:
                l = (line.strip().split('\t')[0]).strip().split(' ')
                l = [int(s) for s in l]
                inp.append(l)
            inp = torch.LongTensor(np.asarray(inp, dtype='int64'))
            samples = generator.conditional_sample(inp)
            # construct the input to the genrator, add zeros before samples and delete the last column
            zeros = torch.zeros((BATCH_SIZE, 1)).type(torch.LongTensor)
            if samples.is_cuda:
                zeros = zeros.cuda()
            inputs = Variable(torch.cat([zeros, inp.data], dim = 1)[:, :-1].contiguous())
            targets = Variable(samples.data).contiguous().view((-1,))
            # calculate the reward
            rewards = rollout.get_reward(samples, 16, discriminator)
            rewards = Variable(torch.Tensor(rewards))
            rewards = torch.exp(rewards).contiguous().view((-1,))
            if opt.cuda:
                rewards = rewards.cuda()
            prob = generator.forward(inputs)
            loss = gen_gan_loss(prob, targets, rewards)
            gen_gan_optm.zero_grad()
            loss.backward()
            gen_gan_optm.step()

        if total_batch % 1 == 0 or total_batch == TOTAL_BATCH - 1:
            generate_samples(generator, BATCH_SIZE, GENERATED_NUM, POSITIVE_FILE, EVAL_FILE)
            eval_iter = GenDataIter(EVAL_FILE, BATCH_SIZE)
            loss = eval_epoch(generator, eval_iter, gen_criterion)
            print('Batch [%d] True Loss: %f' % (total_batch, loss))
        rollout.update_params()

        for _ in range(4):
            generate_samples(generator, BATCH_SIZE, GENERATED_NUM, POSITIVE_FILE, NEGATIVE_FILE)
            dis_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, BATCH_SIZE)
            for _ in range(2):
                loss = train_epoch(discriminator, dis_data_iter, dis_criterion, dis_optimizer)
    
    # Loss in the test data
    test_iter = GenDataIter(TEST_FILE, BATCH_SIZE)
    loss = eval_epoch(generator, test_iter, gen_criterion)
    print('Test Loss: %f' % (loss))

if __name__ == '__main__':
    main()
