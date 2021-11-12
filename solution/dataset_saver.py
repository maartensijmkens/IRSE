import os
import pickle
from torch.utils.data import random_split
from dataset import NUS

# directory to save datasets
directory = 'datasets'

# size of validation set
val_size = 256

# create directory
if not os.path.exists(directory):
    os.makedirs(directory)

# filepaths for datasets
train_path = os.path.join(directory, 'train.pickle')
val_path = os.path.join(directory, 'val.pickle')
test_path = os.path.join(directory, 'test.pickle')

# load datatsets
train_set = NUS(mode='Train')
test_set = NUS(mode='Test')
_, val_set = random_split(test_set, [len(test_set) - val_size, val_size])

# write datasets to file
with open(train_path, 'wb') as f1, open(val_path, 'wb') as f2, open(test_path, 'wb') as f3:
    pickle.dump(train_set, f1)
    pickle.dump(val_set, f2)
    pickle.dump(test_set, f3)
