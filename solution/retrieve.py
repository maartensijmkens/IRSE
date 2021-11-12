import os
import json
import argparse

import torch
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt

from models import ProjectionModel


parser = argparse.ArgumentParser()
parser.add_argument('query', type=str)
parser.add_argument('--exp', type=str, required=True)
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--image_folder', type=str, default=os.path.join('NUS-WIDE-Lite','images'))
args = parser.parse_args()

print(args.query)

config_path = os.path.join('exp', args.exp, 'config.json')
bag_of_words_path = os.path.join('exp', args.exp, 'bag_of_words.pickle')
checkpoint_path = os.path.join('exp', args.exp, 'best.pth')
projections_path = os.path.join('exp', args.exp, 'projections.pth')

# load config
with open(config_path) as f:
    cfg = json.load(f)

# recreate bag of words
with open(bag_of_words_path, 'rb') as f:
    bag_of_words = pickle.load(f)

# select device to retrieve on
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Retrieving on', device)

# load models and move to selected device
checkpoint = torch.load(checkpoint_path)
cap_model = ProjectionModel(cfg['cap_dim'], cfg['hid_dim'], cfg['lat_dim'])
cap_model.load_state_dict(checkpoint['cap_model'])
cap_model.to(device)
cap_model.eval()

# load projections
projections = torch.load(projections_path)
F = projections['img']
filenames = projections['filenames']

# calculate projection of the query
Z = bag_of_words.transform([args.query]).tocoo()
Z = torch.sparse_coo_tensor([Z.row, Z.col], Z.data, Z.shape, dtype=torch.float32)
Z = Z.to(device)
G = cap_model(Z)

# calculate distances and take the best 10
D = torch.cdist(G, F)
top_k = np.take(filenames, torch.argsort(D).cpu())[0,:args.k]

# show k best images
for i, fn in enumerate(top_k):
    im_path = os.path.join(args.image_folder, fn + '.jpg')
    im = cv2.imread(im_path)
    cv2.imshow('image ' + str(i+1), im)
    cv2.waitKey()
    cv2.destroyAllWindows()

plt.show()
