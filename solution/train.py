import os
import json
import datetime
import argparse

import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from models import ProjectionModel
from losses import triplet_loss_0, triplet_loss_1, triplet_loss_2, triplet_loss_3
from metrics import mrr


parser = argparse.ArgumentParser()
parser.add_argument('--loss', choices=['triplet_0', 'triplet_1', 'triplet_2', 'triplet_3'], default='triplet_1')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--margin', type=float, default=0.5)
parser.add_argument('--img_dim', type=int, default=2048)
parser.add_argument('--cap_dim', type=int, default=300)
parser.add_argument('--hid_dim', type=int, default=2048)
parser.add_argument('--lat_dim', type=int, default=512)
parser.add_argument('--exp', type=str, default=datetime.datetime.now().strftime("%d-%m-%Y_%H.%M.%S"))
args = parser.parse_args()


# function for saving train configuration and vocabulary
def save_config(config, directory):
    with open(os.path.join(directory, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

# load train and val dataset
datasets_dir = 'datasets'
train_path = os.path.join(datasets_dir, 'train.pickle')
val_path = os.path.join(datasets_dir, 'val.pickle')

with open(train_path, 'rb') as f1, open(val_path, 'rb') as f2:
    train_set = pickle.load(f1)
    val_set = pickle.load(f2)

train_loader = DataLoader(train_set, args.batch_size, shuffle=True)
val_loader = DataLoader(val_set, len(val_set), shuffle=False)

# create new experiment
print('creating new experiment with name', args.exp)
directory = os.path.join('exp', args.exp)
os.makedirs(directory)
config = vars(args)
save_config(config, directory)

# select device to train on
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Training on', device)

# initialize models and move to selected device
img_model = ProjectionModel(args.img_dim, args.hid_dim, args.lat_dim)
img_model.to(device)
cap_model = ProjectionModel(args.cap_dim, args.hid_dim, args.lat_dim)
cap_model.to(device)

# get model's trainable parameters and create an optimizer for them
parameters = list(img_model.parameters()) + list(cap_model.parameters())
optimizer = SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=0.0005)
scheduler = ExponentialLR(optimizer, gamma=0.9, verbose=True)

# function for saving current state of a model to a checkpoint
def save_checkpoint(img_model, cap_model, directory):
    checkpoint = {'img_model': img_model.state_dict(), 'cap_model': cap_model.state_dict()}
    torch.save(checkpoint, os.path.join(directory, 'best.pth'))


# function for performing one training iteration
def train_step(img_model, cap_model, optimizer, device, train_loader):
    global args

    # set model to training mode and initialize the total loss
    img_model.train()
    cap_model.train()
    total_loss = 0

    # loop over all train batches
    for X, Z, V, _ in tqdm(train_loader):

        # move vectors to selected device and forward them through the model
        X, Z, V = X.to(device), Z.to(device), V.to(device)
        F = img_model(X)
        G = cap_model(Z)

        # calculate the loss of the projections
        if args.loss == 'triplet_0':
            selected_loss = triplet_loss_0
        elif args.loss == 'triplet_1':
            selected_loss = triplet_loss_1
        elif args.loss == 'triplet_2':
            selected_loss = triplet_loss_2
        else:
            selected_loss = triplet_loss_3

        loss = selected_loss(V, F, G, args.margin)
        total_loss += loss.item()

        # perform backwards propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # return the average loss over all train batches
    N = len(train_loader)
    return total_loss / N


# function for performing one validation iteration
def val_step(img_model, cap_model, device, val_loader):
    
    # set the models to eval mode and initialize total mrr
    img_model.eval()
    cap_model.eval()
    total_mrr = 0

    # loop over all val batches
    for X, Z, _, _ in val_loader:

        # move vectors to selected device and forward them through the models
        X, Z = X.to(device), Z.to(device)
        F = img_model(X)
        G = cap_model(Z)

        # add the mrr of this batch to the total
        total_mrr += mrr(F, G)

    # return the average mrr of all val batches
    return total_mrr / len(val_loader)


# plot learning history
def plot_history(loss_history, mrr_history, directory):
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.set_title('train loss')
    ax1.plot(loss_history)
    ax2.set_title('val MRR')
    ax2.plot(mrr_history)
    fig.savefig(os.path.join(directory, 'history.png'))
    plt.close(fig)

train_loss_history = []
val_mrr_history = []
best_val_mrr = 0


# training loop
for epoch in range(1, args.epochs + 1):

    # perform one train and one val step
    train_loss = train_step(img_model, cap_model, optimizer, device, train_loader)
    train_loss_history.append(train_loss)
    val_mrr = val_step(img_model, cap_model, device, val_loader)
    val_mrr_history.append(val_mrr)

    # report train_loss and val_mrr
    GREEN = '\033[92m'
    BLUE = '\033[96m'
    DEFAULT = '\033[0m'
    print(
        'Epoch', epoch, '|| train_loss:', 
        GREEN, round(train_loss, 4), DEFAULT,
        'val_MRR:', GREEN, round(val_mrr, 4), DEFAULT
    )
    plot_history(train_loss_history, val_mrr_history, directory)

    # save models if best val_mrr
    if val_mrr > best_val_mrr:
        best_val_mrr = val_mrr
        save_checkpoint(img_model, cap_model, directory)

    # adjust learning rate
    if epoch % 10 == 0:
        scheduler.step()
