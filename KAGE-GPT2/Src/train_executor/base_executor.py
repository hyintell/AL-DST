import os
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from utils.log_system import logger

class BaseExecutor():
    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader

    def save_checkpoint(self, epoch, batch_id=0, record_best_model=False):
        state = {
            'epoch': epoch,
            'batch_id': batch_id,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        if record_best_model:
            file_name = "model_best.pth.tar"
            path_save_model = os.path.join(self.config.saved_model_path, file_name)
            torch.save(state, path_save_model)
            logger.print('Model Saved:', path_save_model)
        else:
            file_name = "model_{}.pth.tar".format(epoch)
            path_save_model = os.path.join(self.config.saved_model_path, file_name)
            # Save the state
            torch.save(state, path_save_model)
            logger.print('Model Saved:', path_save_model)

            file_name = "model_lastest.pth.tar".format(epoch)
            path_save_model = os.path.join(self.config.saved_model_path, file_name)
            # Save the state
            torch.save(state, path_save_model)
            logger.print('Lastest Model Saved:', path_save_model)


    def local_save_checkpoint(self, save_model_prefix, epoch, round_idx, batch_id=0):
        state = {
            'round': round_idx,
            'epoch': epoch,
            'batch_id': batch_id,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }

        # create the path if don't have
        Path(save_model_prefix).mkdir(parents=True, exist_ok=True)
        checkpointing_path = f'{save_model_prefix}/model_best.pth.tar'
        # file_name = "model_best.pth.tar"
        # path_save_model = os.path.join(self.config.saved_model_path, file_name)

        torch.save(state, checkpointing_path)
        print(f'Model Saved for round {round_idx} at epoch {epoch}:', checkpointing_path)
    
            
    def sagemaker_save_checkpoint(self, save_model_prefix, current_epoch_train_loss, current_epoch_gen_loss, epoch, round_idx=-1, batch_id=0):
        """
        Save checkpoint on Sagemaker
        """
        
        state = {
            'round': round_idx,
            'epoch': epoch,
            'batch_id': batch_id,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'train_loss': current_epoch_train_loss,
            'gen_loss': current_epoch_gen_loss
        }
        
        ##############################
        # create the path if don't have
        Path(save_model_prefix).mkdir(parents=True, exist_ok=True)
        
#         checkpointing_path = f'{save_model_prefix}/checkpoint-epoch-{epoch}.pth.tar'
        checkpointing_path = f'{save_model_prefix}/model_best.pth.tar'
        print(f'Checkpoint Saved: {checkpointing_path}')
        print(f'round: {round_idx}, epoch: {epoch}, current_epoch_train_loss: {current_epoch_train_loss}, current_epoch_gen_loss: {current_epoch_gen_loss}')
        torch.save(state, checkpointing_path)


    def load_saved_model(self, load_model_path=""):
        
        #### load checkpoint from Spot Training
        path_save_model = f'{load_model_path}/model_best.pth.tar'
        print(f'path_save_model: {path_save_model}')
        
        checkpoint = torch.load(path_save_model, map_location='cuda:{}'.format(self.config.gpu_device))
        self.loaded_epoch = int(checkpoint['epoch'])
        self.model.load_state_dict(checkpoint['state_dict'])

        round_idx = int(checkpoint['round'])

        if 'batch_id' in checkpoint.keys():
            batch_id = checkpoint['batch_id']
        else:
            batch_id = 0
        print("Checkpoint loaded successfully from '{}' at (round {} epoch {} batch {})\n"
                .format(path_save_model, round_idx, checkpoint['epoch'], batch_id))
        print(f"Loading checkpoint: round-{round_idx}, epoch-{self.loaded_epoch}")
    

    def load_checkpoint_model(self, load_epoch=-1, load_best_model=False, load_model_path=""):
        print(f'$$$: load_epoch: {load_epoch}, load_best_model: {load_best_model}, load_model_path: {load_model_path}')
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
            self.loaded_epoch = int(checkpoint['epoch'])
            if not load_model_path:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.load_pretrain_weights(checkpoint['state_dict'])
            if not load_model_path:
                # Loading from external model weights, do not load optimizer
                if 'optimizer' in checkpoint.keys():
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    print('< optimizer loaded from checkpoint >')
                if 'scheduler' in checkpoint.keys():
                    self.scheduler.load_state_dict(checkpoint['scheduler'])
                    print('< scheduler loaded from checkpoint >')
            if 'batch_id' in checkpoint.keys():
                batch_id = checkpoint['batch_id']
            else:
                batch_id = 0
            # self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Checkpoint loaded successfully from '{}' at (epoch {} batch {})\n"
                  .format(path_save_model, checkpoint['epoch'], batch_id))

            if load_model_path:
                self.loaded_epoch = 0 # Load model == start from epoch 0
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
            
            
#     def sagemaker_load_checkpoint_model(self, latest_checkpoint_filename, load_epoch=-1, load_best_model=False, load_model_path=""):
#         """
#         Load checkpoint on Sagemaker
#         """
        
#         #### load checkpoint from Spot Training
#         print(f'load_epoch: {load_epoch}, load_best_model: {load_best_model}, load_model_path: {load_model_path}')
        
#         if load_model_path:
#             path_save_model = load_model_path
#         else:
#             if load_best_model:
#                 file_name = "model_best.pth.tar"
#             else:
#                 if load_epoch == -1:
#                     file_name = "model_lastest.pth.tar"
#                 else:
#                     file_name = "model_{}.pth.tar".format(load_epoch)

#             path_save_model = os.path.join(self.config.saved_model_path, file_name)
        
#         print(f'Loading Checkpoint From: {self.config.checkpoint_path}/{latest_checkpoint_filename}')
        
#         path_save_model = f'{self.config.checkpoint_path}/{latest_checkpoint_filename}'
#         checkpoint = torch.load(path_save_model, map_location='cuda:{}'.format(self.config.gpu_device))
#         self.loaded_epoch = int(checkpoint['epoch'])
#         self.model.load_state_dict(checkpoint['state_dict'])
        
#         if 'optimizer' in checkpoint.keys():
#             self.optimizer.load_state_dict(checkpoint['optimizer'])
#             print('< optimizer loaded from checkpoint >')
#         if 'scheduler' in checkpoint.keys():
#             self.scheduler.load_state_dict(checkpoint['scheduler'])
#             print('< scheduler loaded from checkpoint >')
#         if 'batch_id' in checkpoint.keys():
#             batch_id = checkpoint['batch_id']
#         else:
#             batch_id = 0

#         print(f"Checkpoint loaded successfully from '{path_save_model}' at (epoch {checkpoint['epoch']} batch {batch_id})\n")
#         print(f"Loading checkpoint: epoch-{self.loaded_epoch}, train_loss-{checkpoint['train_loss']}, gen_loss-{checkpoint['gen_loss']}")
    
    
    def save_csv(self, df, filename):
        """
        Save selected turn ids and time, open the same csv and append the results to it
        """
        
        ## save selected turn id to csv
        if self.config.ENV == 'sagemaker_remote':
            # Running on Sagemaker training job
        
            save_result_prefix = self.config.output_data_dir
        else:
            # running on local
            save_result_prefix = self.config.saved_model_path
            
        if not os.path.exists(save_result_prefix):
            os.mkdir(save_result_prefix)
            
        save_result_prefix = f'{save_result_prefix}/{filename}'
        
        print(f'Saving {filename}: {save_result_prefix}')
        df.to_csv(save_result_prefix, mode='a', header=not os.path.exists(save_result_prefix), index=False)
            
    
    def load_csv(self):
        """
        Load selected turn csv file
        """
        
        selected_turn_ids = None
        resumed_round = None
        
        ## save selected turn id to csv
        if self.config.ENV == 'sagemaker_remote':
            # Running on Sagemaker training job
        
            save_result_prefix = f'{self.config.selected_turn_path}/selected_turn_id.csv'
        else:
            # running on local
            ## example: /home/ec2-user/SageMaker/KAGE/Experiments/test_entropy/train/saved_model/selected_turn_id.csv
            save_result_prefix = f'{self.config.saved_model_path}/selected_turn_id.csv'
        
        try:
            print(f'Loading csv from: {save_result_prefix}')
            df = pd.read_csv(save_result_prefix)
            selected_turn_ids = df['selected_turn_id'].tolist()
            selected_turn_ids = np.array(selected_turn_ids)
            print(f'###### selected_turn_ids: {len(selected_turn_ids)}  {selected_turn_ids}')
            
            resumed_round = df['round'].max()
            print(f'###### maximum round: {resumed_round}')
            
            # save a copy to the new 'self.config.output_data_dir' path
            if self.config.continue_select_turn == 1:
                self.save_csv(df, 'selected_turn_id.csv')
        except Exception as e:
            print(f'Fail to read csv, {e}')
            pass
        
        return selected_turn_ids, resumed_round
    
        
    def train(self):
        raise NotImplementedError("Train function has not been defined!")


    def load_pretrain_weights(self, pretrained_dict):
        model_dict = self.model.state_dict()
        # 1. filter out unnecessary keys
        ignored_keys = []
        if len(self.config.ignore_pretrained_weights) > 0:
            for key in pretrained_dict.keys():
                for ignore_str in self.config.ignore_pretrained_weights:
                    if ignore_str in key:
                        ignored_keys.append(key)
        print('follwing pretrained weights are ignored', ignored_keys)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and k not in ignored_keys}
        print('Loading pretrained weights', [k for k in pretrained_dict.keys()])
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.model.load_state_dict(model_dict)