import os
import json
import argparse

import pickle
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from models import ProjectionModel
from metrics import mrr


parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, required=True)
args = parser.parse_args()

config_path = os.path.join('exp', args.exp, 'config.json')
bag_of_words_path = os.path.join('exp', args.exp, 'bag_of_words.pickle')
checkpoint_path = os.path.join('exp', args.exp, 'best.pth')

# load config
with open(config_path) as f:
    cfg = json.load(f)

# recreate bag of words
with open(bag_of_words_path, 'rb') as f:
    bag_of_words = pickle.load(f)

# select device to test on
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Testing on', device)

# load models and move to selected device
checkpoint = torch.load(checkpoint_path)

img_model = ProjectionModel(cfg['img_dim'], cfg['hid_dim'], cfg['lat_dim'])
img_model.load_state_dict(checkpoint['img_model'])
img_model.to(device)
img_model.eval()

cap_model = ProjectionModel(cfg['cap_dim'], cfg['hid_dim'], cfg['lat_dim'])
cap_model.load_state_dict(checkpoint['cap_model'])
cap_model.to(device)
cap_model.eval()

# load test dataset
datasets_dir = 'datasets'
test_path = os.path.join(datasets_dir, 'test.pickle')

with open(test_path, 'rb') as f:
    test_set = pickle.load(f)
    test_loader = DataLoader(test_set, 256, shuffle=False)

# initialize total mrr and all projections arrays
total_mrr = 0
all_F = []
all_G = []

# loop over all test batches
for X, Y, _, _ in test_loader:

    # convert captions to bag of words vectors
    Z = bag_of_words.transform(Y).tocoo()
    Z = torch.sparse_coo_tensor([Z.row, Z.col], Z.data, Z.shape, dtype=torch.float32)

    # move vectors to selected device and forward them through the model
    X, Z = X.to(device), Z.to(device)
    F = img_model(X)
    G = cap_model(Z)

    # add the mrr of this batch to the total
    all_F.append(F)
    all_G.append(G)
    total_mrr += mrr(F, G)


# report batch mrr
N = len(test_loader)
print('batch mrr', total_mrr / N)

# save all projections
all_F = torch.cat(all_F)
all_G = torch.cat(all_G)

projections = {'img': all_F, 'cap': all_G, 'filenames': test_set.filenames}
torch.save(projections, os.path.join('exp', args.exp, 'projections.pth'))

batch_size = 128
all_G_loader = DataLoader(all_G, batch_size, shuffle=False)
total_mrr = 0

# calculate mrr
for i, Gi in enumerate(tqdm(all_G_loader)):
    n = Gi.shape[0]
    total_mrr += n * mrr(all_F, Gi, batch_size*i)

N = all_G.shape[0]
print('total mrr', total_mrr / N)
