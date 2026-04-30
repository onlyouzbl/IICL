
# -*- coding: utf-8 -*-
"""
Dataset and dataloader functions
"""

import os
import json
import random
random.seed(1234)
from random import choice
import pickle
from PIL import Image
import torch
from torch.utils.data import Dataset
from utils.utils import get_token_ids, list2Tensors
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import multiprocessing


class Recipe1M(Dataset):


    def __init__(self, root, transform=None, split='train',
                 max_ingrs=20,
                 max_instrs=20,
                 max_length_ingrs=15,
                 max_length_instrs=15,
                 text_only_data=False,
                 training_data_comb="123"):

        #load vocabulary
        self.vocab_inv = pickle.load(open('../data/vocab.pkl', 'rb'))
        self.vocab = {}
        for k, v in self.vocab_inv.items():
            if type(v) != str:
                v = v[0]
            self.vocab[v] = k
        # suffix to load text only samples or paired samples
        suf = '_noimages' if text_only_data else ''
       
        self.data = pickle.load(open(os.path.join(root, 'traindata', split + suf + '.pkl'),
                                     'rb'))
        self.root = root    #DATASET_PATH
        self.ids = list(self.data.keys())    

        self.split = split     # 'train'
        self.transform = transform    

        self.max_ingrs = max_ingrs   
        self.max_instrs = max_instrs  
        self.max_length_ingrs = max_length_ingrs   
        self.max_length_instrs = max_length_instrs  

        self.text_only_data = text_only_data   

        #########
        self.training_data_comb=training_data_comb

    def __getitem__(self, idx):

      
        entry = self.data[self.ids[idx]]

        if not self.text_only_data:
           

            img_name = entry['images'][0]

            
            img = Image.open(os.path.join(self.root, self.split, img_name))
            if self.transform is not None:
                img = self.transform(img)    
        else:
            img = None

        

        title = entry['title']
        
        ingrs = entry['ingredients']
        
        instrs = entry['instructions']

        recipe = []
        instrs_whole = []
        ingrs_whole = []


        title = get_token_ids(title, self.vocab)[:self.max_length_instrs]     #15


        if "1" in self.training_data_comb: 
            recipe.extend(title)
    

        title = torch.Tensor(title)

        
        instrs = [get_token_ids(instr, self.vocab)[:self.max_length_instrs] for instr in instrs[:self.max_instrs]]    #15,20
        for instr in instrs:

            if "3" in self.training_data_comb: 
                recipe.extend(instr)

            instrs_whole.extend(instr)

       
        ingrs = [get_token_ids(ingr, self.vocab)[:self.max_length_ingrs] for ingr in ingrs[:self.max_ingrs]]   #15,20
        for ingr in ingrs:

            if "2" in self.training_data_comb: 
                recipe.extend(ingr)

            
            ingrs_whole.extend(ingr)

        recipe = torch.Tensor(recipe)
        instrs_whole = torch.Tensor(instrs_whole)
        ingrs_whole = torch.Tensor(ingrs_whole)


        return img, title, ingrs_whole, instrs_whole, self.ids[idx], recipe



    def __len__(self):
        return len(self.ids)

    def get_ids(self):
        return self.ids

    def get_vocab(self):
        try:
            return self.vocab_inv
        except:
            return None


def pad_input(input):
    """
    creates a padded tensor to fit the longest sequence in the batch
    """

    if len(input[0].size()) == 1:
        l = [len(elem) for elem in input]
        targets = torch.zeros(len(input), max(l)).long() 
        for i, elem in enumerate(input):
            end = l[i]
            targets[i, :end] = elem[:end]
    else:
        n, l = [], []
        for elem in input:
            n.append(elem.size(0))
            l.append(elem.size(1))
        targets = torch.zeros(len(input), max(n), max(l)).long()    
        for i, elem in enumerate(input):
            targets[i, :n[i], :l[i]] = elem
    return targets


def collate_fn(data):
    
    image, titles, ingrs, instrs, ids, recipes = zip(*data)

    if image[0] is not None:
        # Merge images (from tuple of 3D tensor to 4D tensor).
        image = torch.stack(image, 0)    
    else:
        image = None
    title_targets = pad_input(titles)    
    ingredient_targets = pad_input(ingrs)  
    instruction_targets = pad_input(instrs)  
    recipe_targets = pad_input(recipes)
    


    
    return image, title_targets, ingredient_targets, instruction_targets, ids, recipe_targets


def get_loader(root, batch_size, resize, im_size, augment=True,
               split='train', mode='train',
               drop_last=True,
               text_only_data=False, training_data_comb="123"):
   
    transforms_list = [transforms.Resize((resize))]   

    if mode == 'train' and augment:
        # Image preprocessing, normalization for pretrained resnet
        transforms_list.append(transforms.RandomHorizontalFlip())      
        transforms_list.append(transforms.RandomCrop(im_size))        

    else:
        transforms_list.append(transforms.CenterCrop(im_size))
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Normalize((0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225)))

    transforms_ = transforms.Compose(transforms_list)


    ds = Recipe1M(root, transform=transforms_, split=split,
                  text_only_data=text_only_data, training_data_comb=training_data_comb)


    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=int((multiprocessing.cpu_count())/4),
                        collate_fn=collate_fn, drop_last=drop_last)

    return loader, ds
