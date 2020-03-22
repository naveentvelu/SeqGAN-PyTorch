# import helpers

# TRAIN_DATA_FILE = 'persona.txt'
# TEST_DATA_FILE = 'persona_test.txt'

# POSITIVE_FILE = 'real.data'
# # Generate Vocab and POSITIVE_FILE
# vocab = helpers.generate_vocab(TRAIN_DATA_FILE, TRAIN_DATA_FILE)

# helpers.generate_data(vocab, TRAIN_DATA_FILE, POSITIVE_FILE)

import torch
import random

l1 = [1, 2, 3, 12]
l2 = [4, 5, 6, 11]
l3 = [7, 8, 9, 10]

for i in range(10):
    print(random.sample(l1, 1))