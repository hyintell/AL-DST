import scipy
import os
from pathlib import Path
import numpy as np
import os
import sys
import scipy
import datetime
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.log_system import logger

class BaseEvaluator():
    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader

    def load_checkpoint_model(self, load_epoch=-1, load_best_model=False, load_model_path=""):
        if load_model_path:
            path_save_model = load_model_path
        else:
            if load_best_model:
                file_name = "model_best.pth.tar"
            else:
                if load_epoch == -1:
                    file_name = "model_lastest.pth.tar"
                else:
                    file_name = "model_{}.pth.tar".format(load_epoch)

            path_save_model = os.path.join(self.config.saved_model_path, file_name)

        try:
            logger.print("Loading checkpoint '{}'".format(path_save_model))
            # checkpoint = torch.load(filename)
            checkpoint = torch.load(path_save_model, map_location='cuda:{}'.format(self.config.gpu_device))
            self.model.load_state_dict(checkpoint['state_dict'])
            # self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'batch_id' in checkpoint.keys():
                batch_id = checkpoint['batch_id']
            else:
                batch_id = 0
            if 'epoch' in checkpoint.keys():
                self.loaded_epoch = checkpoint['epoch']
            else:
                self.loaded_epoch = 0
            # self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Checkpoint loaded successfully from '{}' at (epoch {} batch {})\n"
                  .format(path_save_model, checkpoint['epoch'], batch_id))

        except OSError as e:
            self.loaded_epoch = 0
            print(e)
            print("No checkpoint exists from '{}'. Skipping...".format(path_save_model))
            print("**First time to train**")
            
    
    def sagemaker_load_checkpoint_model(self, load_best_model=False, load_model_path=""):
        
        #### load checkpoint from Spot Training
        print(f'load_best_model: {load_best_model}, load_model_path: {load_model_path}')
        
        # load from author-provided epoch
        if load_model_path != '':
            path_save_model = load_model_path
            checkpoint = torch.load(path_save_model, map_location='cuda:{}'.format(self.config.gpu_device))
            self.loaded_epoch = int(checkpoint['epoch'])
            self.model.load_state_dict(checkpoint['state_dict'])
            if 'batch_id' in checkpoint.keys():
                batch_id = checkpoint['batch_id']
            else:
                batch_id = 0
            print("Checkpoint loaded successfully from '{}' at (epoch {} batch {})\n"
                  .format(path_save_model, checkpoint['epoch'], batch_id))
            print(f"Loading checkpoint: epoch-{self.loaded_epoch}, train_loss-{checkpoint['train_loss']}, gen_loss-{checkpoint['gen_loss']}")
        else:
            ########################
            print(f'Loading Checkpoint From: ../checkpoints/checkpoint-epoch-{load_epoch}.pth.tar')

            path_save_model = f'../checkpoints/checkpoint-epoch-{load_epoch}.pth.tar'
            checkpoint = torch.load(path_save_model, map_location='cuda:{}'.format(self.config.gpu_device))
            self.loaded_epoch = int(checkpoint['epoch'])
            self.model.load_state_dict(checkpoint['state_dict'])

    #         if 'optimizer' in checkpoint.keys():
    #             self.optimizer.load_state_dict(checkpoint['optimizer'])
    #             print('< optimizer loaded from checkpoint >')
    #         if 'scheduler' in checkpoint.keys():
    #             self.scheduler.load_state_dict(checkpoint['scheduler'])
    #             print('< scheduler loaded from checkpoint >')
            if 'batch_id' in checkpoint.keys():
                batch_id = checkpoint['batch_id']
            else:
                batch_id = 0

            print("Checkpoint loaded successfully from '{}' at (epoch {} batch {})\n"
                      .format(path_save_model, checkpoint['epoch'], batch_id))
            print(f"Loading checkpoint: epoch-{self.loaded_epoch}, train_loss-{checkpoint['train_loss']}, gen_loss-{checkpoint['gen_loss']}")

    def evaluate(self):
        raise NotImplementedError("Evaluate function has not been defined!")