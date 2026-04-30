# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""
Training script
"""
import os
import random
from collections import defaultdict
import numpy as np
import pickle
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.cuda.amp import GradScaler, autocast
import gc
from torch.utils.tensorboard import SummaryWriter
from utils.utils import *
from utils.loss import TripletLoss, MILNCELoss
from dataset import get_loader
from config import get_args
from models import get_model
from eval import computeAverageMetrics
import logging
from pathlib import Path
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAP_LOC = None if torch.cuda.is_available() else 'cpu'



def trainIter(args, split, loader, model, optimizer,
              loss_function, loss_CL_function, scaler, metrics_epoch):
    with autocast():     # pytorch混精度训练AutoCast与GradScaler ######################

        '''
                img: output a tensor like (128,3,224,224)     or    None
                title: output a tensor like (128,15)        torch.Size([256, 15])
                ingrs: output a tensor like (128,18,15)     torch.Size([256, 20, 15])
                instrs: output a tensor like (128,19,15)    torch.Size([256, 20, 15])
                _: ['000075604a',...]
        '''
        img, title, ingrs, instrs, _, recipe = next(loader)


        img = img.to(device) if img is not None else None
        title = title.to(device)
        ingrs = ingrs.to(device)
        instrs = instrs.to(device)
        recipe = recipe.to(device)

        if split == 'val':
            with torch.no_grad():
                img_feat, recipe_feat, proj_feats, = model(img, title, ingrs, instrs, recipe)
        else:
            out = model(img, title, ingrs, instrs, recipe,
                        freeze_backbone=args.freeze_backbone)         #False

            img_feat, recipe_feat, proj_feats = out
            
            

            
            
       
        loss_recipe, loss_paired = 0, 0

        ######interCL######
        loss_interCL = 0
        loss_titleCL, loss_ingrsCL, loss_instrsCL, loss_recipeCL = 0, 0, 0, 0
        
        #######intraCL######
        loss_intraCL = 0
        loss_intraCL_title_ingrs, loss_intraCL_title_instrs, loss_intraCL_title_recipe, \
        loss_intraCL_ingrs_instrs, loss_intraCL_ingrs_recipe, loss_intraCL_instrs_recipe = 0, 0, 0, 0, 0, 0

        if args.recipe_loss_weight > 0:     
           
            c = 0
            names = ['title', 'ingredients', 'instructions']
            for raw_name in names:
                q = proj_feats['raw'][raw_name]
                for proj_name in names:
                    if proj_name != raw_name:
                       
                        v = proj_feats[proj_name][raw_name]
                        loss_recipe += loss_function(q, v)
                        c += 1
            loss_recipe /= c

            loss_recipe = args.recipe_loss_weight*loss_recipe

            metrics_epoch['loss_recipe'].append(loss_recipe.item())


        #######Compute intraCL loss
        if args.intraCL_loss_weight > 0:
            d = 0
            if args.intraCL_title_ingrs_loss_weight > 0:
                intraCL_title_ingrs_scores = torch.einsum("md,nd->mn", proj_feats['raw']['title'],
                                                          proj_feats['raw']['ingredients'])
                loss_intraCL_title_ingrs = loss_CL_function(intraCL_title_ingrs_scores)
                loss_intraCL_title_ingrs = args.intraCL_title_ingrs_loss_weight * loss_intraCL_title_ingrs
                metrics_epoch['loss_intraCL_title_ingrs'].append(loss_intraCL_title_ingrs.item())
                d += 1
            if args.intraCL_title_instrs_loss_weight > 0:
                intraCL_title_instrs_scores = torch.einsum("md,nd->mn", proj_feats['raw']['title'],
                                                          proj_feats['raw']['instructions'])
                loss_intraCL_title_instrs = loss_CL_function(intraCL_title_instrs_scores)
                loss_intraCL_title_instrs = args.intraCL_title_instrs_loss_weight * loss_intraCL_title_instrs
                metrics_epoch['loss_intraCL_title_instrs'].append(loss_intraCL_title_instrs.item())
                d += 1
            if args.intraCL_title_recipe_loss_weight > 0:
                intraCL_title_recipe_scores = torch.einsum("md,nd->mn", proj_feats['raw']['title'],
                                                          recipe_feat)
                loss_intraCL_title_recipe = loss_CL_function(intraCL_title_recipe_scores)
                loss_intraCL_title_recipe = args.intraCL_title_recipe_loss_weight * loss_intraCL_title_recipe
                metrics_epoch['loss_intraCL_title_recipe'].append(loss_intraCL_title_recipe.item())
                d += 1
            if args.intraCL_ingrs_instrs_loss_weight > 0:
                intraCL_ingrs_instrs_scores = torch.einsum("md,nd->mn", proj_feats['raw']['ingredients'],
                                                          proj_feats['raw']['instructions'])
                loss_intraCL_ingrs_instrs = loss_CL_function(intraCL_ingrs_instrs_scores)
                loss_intraCL_ingrs_instrs = args.intraCL_ingrs_instrs_loss_weight * loss_intraCL_ingrs_instrs
                metrics_epoch['loss_intraCL_ingrs_instrs'].append(loss_intraCL_ingrs_instrs.item())
                d += 1
            if args.intraCL_ingrs_recipe_loss_weight > 0:
                intraCL_ingrs_recipe_scores = torch.einsum("md,nd->mn", proj_feats['raw']['ingredients'],
                                                          recipe_feat)
                loss_intraCL_ingrs_recipe = loss_CL_function(intraCL_ingrs_recipe_scores)
                loss_intraCL_ingrs_recipe = args.intraCL_ingrs_recipe_loss_weight * loss_intraCL_ingrs_recipe
                metrics_epoch['loss_intraCL_ingrs_recipe'].append(loss_intraCL_ingrs_recipe.item())
                d += 1
            if args.intraCL_instrs_recipe_loss_weight > 0:
                intraCL_instrs_recipe_scores = torch.einsum("md,nd->mn", proj_feats['raw']['instructions'],
                                                          recipe_feat)
                loss_intraCL_instrs_recipe = loss_CL_function(intraCL_instrs_recipe_scores)
                loss_intraCL_instrs_recipe = args.intraCL_instrs_recipe_loss_weight * loss_intraCL_instrs_recipe
                metrics_epoch['loss_intraCL_instrs_recipe'].append(loss_intraCL_instrs_recipe.item())
                d += 1

            loss_intraCL = loss_intraCL_title_ingrs + loss_intraCL_title_instrs + loss_intraCL_title_recipe + \
                           loss_intraCL_ingrs_instrs + loss_intraCL_ingrs_recipe + loss_intraCL_instrs_recipe
            loss_intraCL /= d
            loss_intraCL = args.intraCL_loss_weight * loss_intraCL
            metrics_epoch['loss_intraCL'].append(loss_intraCL.item())


        if img is not None:

    
            loss_paired = loss_function(img_feat, recipe_feat)
            metrics_epoch['loss_paired'].append(loss_paired.item())

            #######Compute interCL loss
            if args.interCL_loss_weight > 0:
                e = 0
                if args.titleCL_loss_weight > 0:
                    img_title_scores = torch.einsum("md,nd->mn", img_feat, proj_feats['raw']['title'])
                    loss_titleCL = loss_CL_function(img_title_scores)
                    loss_titleCL = args.titleCL_loss_weight * loss_titleCL
                    metrics_epoch['loss_titleCL'].append(loss_titleCL.item())
                    e += 1

                if args.ingrsCL_loss_weight > 0:
                    img_ingrs_scores = torch.einsum("md,nd->mn", img_feat, proj_feats['raw']['ingredients'])
                    loss_ingrsCL = loss_CL_function(img_ingrs_scores)
                    loss_ingrsCL = args.ingrsCL_loss_weight * loss_ingrsCL
                    metrics_epoch['loss_ingrsCL'].append(loss_ingrsCL.item())
                    e += 1

                if args.instrsCL_loss_weight > 0:
                    img_instrs_scores = torch.einsum("md,nd->mn", img_feat, proj_feats['raw']['instructions'])
                    loss_instrsCL = loss_CL_function(img_instrs_scores)
                    loss_instrsCL = args.instrsCL_loss_weight * loss_instrsCL
                    metrics_epoch['loss_instrsCL'].append(loss_instrsCL.item())
                    e += 1

                if args.recipeCL_loss_weight > 0:
                    img_recipe_scores = torch.einsum("md,nd->mn", img_feat, recipe_feat)
                    loss_recipeCL = loss_CL_function(img_recipe_scores)
                    loss_recipeCL = args.recipeCL_loss_weight * loss_recipeCL
                    metrics_epoch['loss_recipeCL'].append(loss_recipeCL.item())
                    e += 1

                loss_interCL = loss_titleCL + loss_ingrsCL + loss_instrsCL + loss_recipeCL
                loss_interCL /= e
                loss_interCL = args.interCL_loss_weight * loss_interCL
                metrics_epoch['loss_interCL'].append(loss_interCL.item())

        loss = loss_paired + loss_recipe + loss_intraCL + loss_interCL
        metrics_epoch['loss'].append(loss.item())

   

    if split == 'train':
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    if img_feat is not None:
        img_feat = img_feat.cpu().detach().numpy()
    recipe_feat = recipe_feat.cpu().detach().numpy()

    return img_feat, recipe_feat, metrics_epoch


def train(args):
    
    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        log_filename = time.strftime("%Y-%m-%d-%H-%M-%S.log", time.localtime())
        log_filename = os.path.join(args.log_dir, "{}_{}".format(args.model_name, log_filename))
    else:
        log_filename = None
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.info(str(args))
    
    checkpoints_dir = os.path.join(args.save_dir, args.model_name)   # saved_checkpoints/model
    make_dir(checkpoints_dir)
    logging.info('model_checkpoints_dir: {}'.format(checkpoints_dir))
    

    if args.tensorboard:
        logger = SummaryWriter(checkpoints_dir)

    loaders = {}

    if args.resume_from != '':
        print("Resuming from checkpoint: ", args.resume_from)
        logging.info("Resuming from checkpoint: {}".format(args.resume_from))
        # update arguments when loading model
        vars_to_replace = ['batch_size', 'tensorboard',
                           'model_name', 'lr',
                           'scale_lr', 'freeze_backbone',
                           'load_optimizer']
        store_dict = {}
        for var in vars_to_replace:
            store_dict[var] = getattr(args, var)

        resume_path = os.path.join(args.save_dir, args.resume_from)
        args, model_dict, optim_dict = load_checkpoint(resume_path,
                                                       'curr', MAP_LOC,
                                                       store_dict)

        # load current state of training
        curr_epoch = args.curr_epoch
        best_loss = args.best_loss

        for var in vars_to_replace:
            setattr(args, var, store_dict[var])
    else:
        curr_epoch = 0
        best_loss = np.inf    #np.inf means +∞, there is no exact value, the type is floating point
        model_dict, optim_dict = None, None

    for split in ['train', 'val']:

        loader, dataset = get_loader(args.root, args.batch_size, args.resize,
                                     args.imsize,
                                     augment=True,
                                     split=split, mode=split,
                                     text_only_data=False, training_data_comb=args.training_data_comb)
        loaders[split] = loader

    # create dataloader for training samples without images
    use_extra_data = True

    if (args.recipe_loss_weight > 0 and use_extra_data) or (args.intraCL_loss_weight > 0 and use_extra_data) :       
        print("use extra data with intra CL")
        loader_textonly, _ = get_loader(args.root, args.batch_size*2,     
                                        args.resize,     
                                        args.imsize,    
                                        augment=True,
                                        split='train', mode='train',
                                        text_only_data=True, training_data_comb=args.training_data_comb)

    vocab_size = len(dataset.get_vocab())         
    model = get_model(args, vocab_size)
    
    params_backbone = list(model.image_encoder.backbone.parameters())     
    # note: delete "+ list(model.merger_recipe.parameters()) \"
    params_fc = list(model.image_encoder.fc.parameters()) \
                + list(model.text_encoder.parameters()) \
                + list(model.projector_recipes.parameters())
    
    N_recipe_enc = count_parameters(model.text_encoder)
    N_image_enc = count_parameters(model.image_encoder)
    print("recipe encoder", N_recipe_enc)
    print("image encoder", N_image_enc)
    logging.info("recipe encoder: {}".format(N_recipe_enc))
    logging.info("image encoder: {}".format(N_image_enc))

    optimizer = get_optimizer(params_fc,
                              params_backbone,
                              args.lr, args.scale_lr, args.wd,    
                              freeze_backbone=args.freeze_backbone)   

    if model_dict is not None:
        model.load_state_dict(model_dict)
        if args.load_optimizer:
            try:
                optimizer.load_state_dict(optim_dict)
            except:
                print("Could not load optimizer state. Using default initialization...")

    ngpus = 2   
    if device != 'cpu' and torch.cuda.device_count() > 1:
        ngpus = torch.cuda.device_count()   

        model = nn.DataParallel(model)
    
    print("####gpu used:", ngpus)

    model =model.to(device)
    
    if device != 'cpu':
        cudnn.benchmark = True

    scheduler = get_scheduler(args, optimizer)

    loss_function = TripletLoss(margin=args.margin)    #0.3

    loss_CL_function = MILNCELoss(reduction='mean')
    # training loop
    wait = 0

    scaler = GradScaler()    
    best_epoch = 0

    for epoch in range(curr_epoch, args.n_epochs):    #100

        for split in ['train', 'val']:
            if split == 'train':
                model.train()
            else:
                model.eval()

            metrics_epoch = defaultdict(list)

            total_step = len(loaders[split])
            loader = iter(loaders[split])     

            if (args.recipe_loss_weight > 0 and use_extra_data) or (args.intraCL_loss_weight > 0 and use_extra_data):    # 1.0 and True
                iterator_textonly = iter(loader_textonly)

            img_feats, recipe_feats = None, None

            emult = 2 if (args.recipe_loss_weight > 0 and use_extra_data and split == 'train') or (args.intraCL_loss_weight > 0 and use_extra_data and split == 'train') else 1

            for i in range(total_step*emult):

                if i%2 == 0 and emult == 2:
                    this_iter_loader = iterator_textonly
                else:
                    this_iter_loader = loader

                optimizer.zero_grad()
                model.zero_grad()
               
                img_feat, recipe_feat, metrics_epoch = trainIter(args, split,
                                                                 this_iter_loader,
                                                                 model, optimizer,
                                                                 loss_function,
                                                                 loss_CL_function,
                                                                 scaler,
                                                                 metrics_epoch)

                if img_feat is not None:
                    if img_feats is not None:
                        img_feats = np.vstack((img_feats, img_feat))
                        recipe_feats = np.vstack((recipe_feats, recipe_feat))
                    else:
                        img_feats = img_feat
                        recipe_feats = recipe_feat

                if not args.tensorboard and i != 0 and i % args.log_every == 0:    #False 10
                    # log metrics to stdout every few iterations
                    
                    avg_metrics = {k: np.mean(v) for k, v in metrics_epoch.items() if v}
                    text_ = "split: {:s}, epoch [{:d}/{:d}], step [{:d}/{:d}]"
                    values = [split, epoch, args.n_epochs, i, total_step]
                    for k, v in avg_metrics.items():
                        text_ += ", " + k + ": {:.4f}"
                        values.append(v)
                    str_ = text_.format(*values)
                    print(str_)
                    logging.info(str_)

            # computes retrieval metrics (average of 10 runs on 1k sized rankings)
            retrieval_metrics = computeAverageMetrics(img_feats, recipe_feats,
                                                      1000, 10, forceorder=True)
            '''
                    retrieval_metrics:
                        {
                            'medr': [10 values], 'recall_1': , 'recall_5': , 'recall_10':
                        }
                '''

            for k, v in retrieval_metrics.items():
                metrics_epoch[k] = v

            avg_metrics = {k: np.mean(v) for k, v in metrics_epoch.items() if v}
            # log to stdout at the end of the epoch (for both train and val splits)
            if not args.tensorboard:      # False
                text_ = "AVG. split: {:s}, epoch [{:d}/{:d}]"
                values = [split, epoch, args.n_epochs]
                for k, v in avg_metrics.items():
                    text_ += ", " + k + ": {:.4f}"
                    values.append(v)
                str_ = text_.format(*values)
                print(str_)
                logging.info(str_)

            # log to tensorboard at the end of one epoch
            if args.tensorboard:
                # 1. Log scalar values (scalar summary)
                for k, v in metrics_epoch.items():
                    logger.add_scalar('{}/{}'.format(split, k), np.mean(v), epoch)


        if args.es_metric.startswith('recall'):   
        else:
            mult = 1

        curr_loss = np.mean(metrics_epoch[args.es_metric])

        if args.lr_decay_factor != -1:     
            if args.scheduler_name == 'ReduceLROnPlateau':
                scheduler.step(curr_loss)
            else:
                scheduler.step()

        if curr_loss*mult < best_loss:
            if not args.tensorboard:    
                print("Updating best checkpoint")
                logging.info("Updating best checkpoint")
                logging.info("Updating best epoch: {}".format(epoch))
            save_model(model, optimizer, 'best', checkpoints_dir, ngpus)   #saved_checkpoints/model
            best_loss = curr_loss*mult
            best_epoch = epoch

            wait = 0
        else:
            wait += 1

        save_model(model, optimizer, 'curr', checkpoints_dir, ngpus)
        args.best_loss = best_loss
        args.curr_epoch = epoch
        pickle.dump(args, open(os.path.join(checkpoints_dir,
                                            'args.pkl'), 'wb'))

        if wait == args.patience:
            break
    
    print("Final best epoch is:", best_epoch)
    logging.info("Final best epoch is: {}".format(best_epoch))

    if args.tensorboard:
        logger.close()


def main():
    args = get_args()
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    train(args)


if __name__ == "__main__":
    main()
