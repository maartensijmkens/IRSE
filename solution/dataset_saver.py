import os
import pickle
from dataset import NUS

# directory to save datasets
directory = 'datasets'

# create directory
if not os.path.exists(directory):
    os.makedirs(directory)

# filepaths for datasets
train_path = os.path.join(directory, 'train.pickle')
test_path = os.path.join(directory, 'test.pickle')
val_path = os.path.join(directory, 'val.pickle')

# filenames of subtest set
with open(os.path.join('NUS-WIDE-Lite','image list','NUS_Test_Subset.txt')) as f:
    val_filenames = f.readlines()
    val_filenames = list(map(lambda fn: fn.split('.')[0], val_filenames))

# load datatsets
train_set = NUS(mode='Train')
test_set = NUS(mode='Test')

# write datasets to file
with open(train_path, 'wb') as f1, open(test_path, 'wb') as f2, open(val_path, 'wb') as f3:
    pickle.dump(train_set, f1)
    pickle.dump(test_set, f2)
    test_set.filenames = val_filenames
    pickle.dump(test_set, f3)
