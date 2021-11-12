import csv
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import FastText
from nltk.tokenize import word_tokenize

class NUS(Dataset):
    def __init__(self, mode, root='NUS-WIDE-Lite/'):
        super().__init__()
        assert mode in ['Train', 'Test']
        print('Loading {} set...'.format(mode))
        
        self.mode = mode
        self.root = root
        
        # Fasttext embeddings
        self.text_embedding = FastText('simple')
        
        # Find all the image categories
        self.unique_categories = self.get_categories()
        
        print('Reading the filenames...')
        self.filenames = self.get_names()
        print('Reading the captions...')
        self.captions = self.get_captions(self.root + 'All_Tags.txt')
        print('Reading ground-truth descriptions...')
        self.groundtruth = self.get_groundtruth()
        print('Reading the image features...')
        self.image_features = self.get_image_features(self.root + 'images/image_features.csv')
        print('')
        
    def __len__(self):
        return len(self.filenames)
    
    
    def __getitem__(self, index):
        # Take the filename.
        filename = self.filenames[index]
        caption_embeddings = self.create_fasttext_embeddings(self.captions[filename.split('_')[1]])

        return self.image_features[filename], caption_embeddings, self.groundtruth[index], index

    
    def get_categories(self):
        unique_categories = []
        # Train set
        path_train = self.root + 'image list/Train_imageOutPutFileList.txt'
        with open(path_train, 'r') as file:
            for name in file:
                tokens = name.split('\\')
                if tokens[0] not in unique_categories:
                    unique_categories.append(tokens[0])
        # Test set
        path_test = self.root + 'image list/Test_imageOutPutFileList.txt'
        with open(path_test, 'r') as file:
            for name in file:
                tokens = name.split('\\')
                if tokens[0] not in unique_categories:
                    unique_categories.append(tokens[0]) 
        # Sort the categories alphabetically 
        unique_categories.sort()
        
        return unique_categories
        
    
    def get_names(self):
        # Get the appropriate names depending on the split used (train, test)
        filenames = []
        path = self.root + 'image list/' + self.mode + '_imageOutPutFileList.txt'
        with open(path, 'r') as file:
            for name in file:
                tokens = name[:-1].split('\\') # discard the '\n' at the end
                filenames.append(tokens[1].split('.')[0])   # remove '.jpg'
                
        return filenames
    
    
    def get_captions(self, path):
        all_tags = open(path, "r", encoding='utf-8')
        captions_dict = {}
        for c in all_tags:
            tokens = c.split('      ')
            if len(tokens) > 1:
                if tokens[0] not in captions_dict.keys():
                    captions_dict[tokens[0]] = tokens[1][:-1]
                    
        return captions_dict
    
    
    def get_groundtruth(self):
        files = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(self.root + 'NUS-WIDE-Lite_groundtruth/'):
            for file in f:
                if self.mode + '.txt' in file and file != 'Lite_GT_Train.txt':
                    files.append(os.path.join(r, file))
        
        # Sort the files because we want the concepts to be sorted.
        files.sort()
        
        # Initialize a tensor with the correct dimensions 
        # in order to create the ground truth matrix.
        # train set: 27807 images, test set: 27808 images and 81 concepts.
        if self.mode == 'Train':
            ground_truth = torch.empty(27807, 81) 
        else:
            ground_truth = torch.empty(27808, 81) 
        
        # Iterate over the files and read the values.
        for i, f in enumerate(files):
            gt_file = open(f, "r")
            for j, gt in enumerate(gt_file):
                ground_truth[j][i] = int(gt[:-1])
        
        return ground_truth
                    
    
    def get_image_features(self, path):
        image_names = []
        image_features = []
        with open(path, mode='r') as file:
            csv_reader = csv.reader(file, delimiter=' ')
            for i, line in enumerate(csv_reader):
                if len(line) > 0:
                    name = line.pop(0)[:-4]
                    if name in self.filenames:
                        image_names.append(name)
                        image_features.append(torch.tensor([float(elem) for elem in line]))
        image_dict = {image_names[i]: features for i, features in enumerate(image_features)}
        return image_dict
    
    def create_fasttext_embeddings(self, caption):
        # Tokenize the caption.
        tokens = word_tokenize(caption)   
        word_feats = self.text_embedding.get_vecs_by_tokens(tokens)
        # Take the mean embedding vector as the caption representation.
        caption_emb = torch.mean(word_feats, 0)
        # Alternatively, you can take the summation.
        #caption_emb = torch.sum(word_feats, 0)
        return caption_emb
