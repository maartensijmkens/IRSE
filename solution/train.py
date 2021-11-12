import os
import json
import datetime
import argparse

import pickle
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from models import ProjectionModel
from losses import assignment_loss, modified_loss
from metrics import mrr


parser = argparse.ArgumentParser()
parser.add_argument('--loss', choices=['assignment', 'modified'], default='modified')
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--margin', type=float, default=0.5)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--beta', type=float, default=0.0)
parser.add_argument('--img_dim', type=int, default=2048)
parser.add_argument('--cap_dim', type=int, default=90000)
parser.add_argument('--hid_dim', type=int, default=512)
parser.add_argument('--lat_dim', type=int, default=32)
parser.add_argument('--exp', type=str, default=datetime.datetime.now().strftime("%d-%m-%Y_%H.%M.%S"))
args = parser.parse_args()


# function for saving train configuration and vocabulary
def save_config(config, bag_of_words, directory):
    with open(os.path.join(directory, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    with open(os.path.join(directory, 'vocabulary.json'), 'w') as f:
        json.dump(list(bag_of_words.vocabulary_), f, indent=4)
    with open(os.path.join(directory, 'bag_of_words.pickle'), 'wb') as f:
        pickle.dump(bag_of_words, f)

# load train and val dataset
datasets_dir = 'datasets'
train_path = os.path.join(datasets_dir, 'train.pickle')
val_path = os.path.join(datasets_dir, 'val.pickle')

with open(train_path, 'rb') as f1, open(val_path, 'rb') as f2:
    train_set = pickle.load(f1)
    val_set = pickle.load(f2)

train_loader = DataLoader(train_set, args.batch_size, shuffle=True)
val_loader = DataLoader(val_set, len(val_set), shuffle=False)

# create bag of words 
bag_of_words = CountVectorizer(max_features=args.cap_dim, strip_accents='ascii')
train_captions = [train_set.captions[fn.split('_')[1]] for fn in train_set.filenames]
bag_of_words.fit(train_captions)

# create new experiment
print('creating new experiment with name', args.exp)
directory = os.path.join('exp', args.exp)
os.makedirs(directory)
config = vars(args)
save_config(config, bag_of_words, directory)

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
scheduler = ExponentialLR(optimizer, gamma=0.9)

# function for saving current state of a model to a checkpoint
def save_checkpoint(img_model, cap_model, directory):
    checkpoint = {'img_model': img_model.state_dict(), 'cap_model': cap_model.state_dict()}
    torch.save(checkpoint, os.path.join(directory, 'best.pth'))


# function for performing one training iteration
def train_step(img_model, cap_model, optimizer, device, train_loader, bag_of_words):
    global args

    # set model to training mode and initialize the total loss
    img_model.train()
    cap_model.train()
    total_loss = 0
    total_t1 = 0
    total_t2 = 0

    # loop over all train batches
    for X, Y, V, _ in tqdm(train_loader):

        # convert captions to bag of words vectors
        Z = bag_of_words.transform(Y).tocoo()
        Z = torch.sparse_coo_tensor([Z.row, Z.col], Z.data, Z.shape, dtype=torch.float32)
        
        # move vectors to selected device and forward them through the model
        X, Z, V = X.to(device), Z.to(device), V.to(device)
        F = img_model(X)
        G = cap_model(Z)

        # calculate the loss of the projections
        if args.loss == 'modified':
            selected_loss = modified_loss
        else:
            selected_loss = assignment_loss

        loss_it, t1_it, t2_it = selected_loss(V, F, G, args.margin, args.alpha)
        loss_i, t1_i, t2_i = 0, 0, 0
        loss_t, t1_t, t2_t = 0, 0, 0

        if args.beta > 0:
            loss_i, t1_i, t2_i = selected_loss(V, F, F, args.margin, args.alpha)
            loss_t, t1_t, t2_t = selected_loss(V, G, G, args.margin, args.alpha)

        loss = (1-args.beta) * loss_it + args.beta/2 * loss_i + args.beta/2 * loss_t
        t1 = (1-args.beta) * t1_it + args.beta/2 * t1_i + args.beta/2 * t1_t
        t2 = (1-args.beta) * t2_it + args.beta/2 * t2_i + args.beta/2 * t2_t

        total_loss += loss.item()
        total_t1 += t1.item()
        total_t2 += t2.item()

        # perform backwards propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # return the average loss over all train batches
    N = len(train_loader)
    return total_loss / N, total_t1 / N, total_t2 / N


# function for performing one validation iteration
def val_step(img_model, cap_model, device, val_loader, bag_of_words):
    
    # set the models to eval mode and initialize total mrr
    img_model.eval()
    cap_model.eval()
    total_mrr = 0

    # loop over all val batches
    for X, Y, _, _ in val_loader:

        # convert captions to bag of words vectors
        Z = bag_of_words.transform(Y).tocoo()
        Z = torch.sparse_coo_tensor([Z.row, Z.col], Z.data, Z.shape, dtype=torch.float32)

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
    train_loss, t1, t2 = train_step(img_model, cap_model, optimizer, device, train_loader, bag_of_words)
    train_loss_history.append(train_loss)
    val_mrr = val_step(img_model, cap_model, device, val_loader, bag_of_words)
    val_mrr_history.append(val_mrr)

    # report train_loss and val_mrr
    GREEN = '\033[92m'
    BLUE = '\033[96m'
    DEFAULT = '\033[0m'
    print(
        'Epoch', epoch, '|| train_loss:', 
        GREEN, round(train_loss, 4), DEFAULT, '=',
        BLUE, round(t1, 4), DEFAULT, '+', BLUE, round(t2 ,4), DEFAULT, 
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
        