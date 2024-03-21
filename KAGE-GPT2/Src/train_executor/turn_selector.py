"""
The code is adapted from:
KAGE_executer.py: Training code for the paper:
Lin, W., Tseng, B. H., & Byrne, B. (2021). Knowledge-Aware Graph-Enhanced GPT-2 for Dialogue State Tracking. EMNLP 2021.
https://arxiv.org/abs/2104.04466v3
"""

__author__ = "Weizhe Lin"
__copyright__ = "Copyright 2021, Weizhe Lin"
__version__ = "1.0.0"
__email__ = "wl356@cam.ac.uk"
__status__ = "Published for Github"


import time
import pandas as pd
import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import scipy
import datetime
import json
import random
import pickle
import shutil
from pathlib import Path
from functools import partial
from easydict import EasyDict
from utils.dirs import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

import math
from train_executor.base_executor import BaseExecutor
from utils.log_system import logger

# Customize
from transformers import GPT2TokenizerFast, GPT2Config, get_linear_schedule_with_warmup

from torch.nn.modules.loss import CrossEntropyLoss
from utils.metrics_manager import MetricsManager
from utils.util_dst import *
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer


class TurnSelector(BaseExecutor):
    def __init__(self, config, data_loader):
        BaseExecutor.__init__(self, config, data_loader)
        self.data_loader = data_loader
        self.model = None
        
        # get train dialogue numbers and ids
        dst_f = self.config.data_loader.additional.train_data_path
        _, _, _, train_unlabelled_dialogue_id2turns, train_unlabelled_dialogue_id_list, dialogue_id2len = iterate_dst_file(dst_f)
        
        if self.config.continue_select_turn != 1:
            # save selected ids
            df = pd.DataFrame(dialogue_id2len.items(), columns=['dialogue_id', 'total_turns'])
            df = df.sort_values('dialogue_id')
            self.save_csv(df, 'train_dialogue2len.csv')
#             df.to_csv('./mwz21_train_dialogue2len.csv', index=False)

        # sort by name
        train_unlabelled_dialogue_id_list.sort()
        self.train_unlabelled_dialogue_id_list = np.array(train_unlabelled_dialogue_id_list)
        
#         ## TESTING: select only partial 50 dialogues for testing
#         self.train_unlabelled_dialogue_id_list = np.random.choice(self.train_unlabelled_dialogue_id_list, 300, replace=False)
        
#         print(f'Select partial dialgoues: {len(self.train_unlabelled_dialogue_id_list)} {self.train_unlabelled_dialogue_id_list}')
        
        # we have budget, then use the budget to train an initial model first
        if self.config.budget > 0:
            # randomly sample dialogues from the unlabelled pool
            self.train_labelled_dialogue_id_list = np.random.choice(self.train_unlabelled_dialogue_id_list, self.config.budget, replace=False)
            
            if self.config.budget_random_turn != 0:
                # randomly sample a turn from a dialogue
                print('Budget training: using random turn of each dialogue')
                self.train_labelled_turn_id_list = [
                    np.random.choice(train_unlabelled_dialogue_id2turns[dia_id], 1, replace=False)[0]['example_idx'] 
                    for dia_id in self.train_labelled_dialogue_id_list
                ]
            else:
                # use last turn from each dialogue 
                print('Budget training: using the last turn of each dialogue')
                self.train_labelled_turn_id_list = [
                    train_unlabelled_dialogue_id2turns[dia_id][-1]['example_idx']
                    for dia_id in self.train_labelled_dialogue_id_list
                ]
            
#             self.train_labelled_turn_id_list = np.array(self.train_labelled_turn_id_list)
            
            # update the unlabelled pool
            self.train_unlabelled_dialogue_id_list = np.setdiff1d(self.train_unlabelled_dialogue_id_list, self.train_labelled_dialogue_id_list)
            
            # save selected ids
            df = pd.DataFrame({
                'round': [-1] * len(self.train_labelled_turn_id_list), # -1 means Budget training stage
                'selected_turn_id': self.train_labelled_turn_id_list,
                'max_entropy': 0,
                'select_turn_per_round_time': [0] * len(self.train_labelled_turn_id_list)
            })
            self.save_csv(df, 'selected_turn_id.csv')
            
        else:
            print('No Budget training: use the random initialized model')
            self.train_labelled_dialogue_id_list = []
            self.train_labelled_turn_id_list = []
        
#         sys.exit(0)

        # Domain slot information from data loader wrapper
        self.value_id2text = data_loader.value_id2text
        self.value_text2id = data_loader.value_text2id
        self.ds_list = data_loader.ds_list
        self.ds_text2id = data_loader.ds_text2id
        self.ontology_value_list = data_loader.ontology_value_list
        self.ontology_value_text2id = data_loader.ontology_value_text2id
        self.ontology_value_id2text = data_loader.ontology_value_id2text
        
        # Create data loaders
        if self.config.data_loader.dummy_dataloader:
            self.load_num = 100
            print(f'Dummy loader, only {self.load_num} examples')
        else:
            self.load_num = -1
        

        # Save tokenizer for later use
        if self.config.ENV == 'sagemaker_remote':
            # running on Sagemaker training job
            if self.config.use_spot_instances:
                # if use Spot training, save the checkpoint to opt/ml/checkpoint
                tokenizer_path = os.path.join(self.config.selected_turn_path, 'tokenizer')
                self.load_model_path = f'{self.config.selected_turn_path}/model_best.pth.tar'
            else:
                # not using Spot training, save the checkpoint to /opt/ml/output/non_spot_checkpoints
                tokenizer_path = os.path.join(self.config.selected_turn_path, 'tokenizer')
                self.load_model_path = f'{self.config.selected_turn_path}/model_best.pth.tar'
        else:
            # running on local
            # print(f'Tokenizer length: {len(self.tokenizer)}')
            tokenizer_path = os.path.join(self.config.saved_model_path, 'tokenizer')
            # tokenizer_path = '../selected_turns/tokenizer'
            self.load_model_path = f'{self.config.saved_model_path}/model_best.pth.tar'

        


        # use loaded tokenizer if continue_select_turn == 1
        if self.config.continue_select_turn == 1:
            # tokenizer_path = os.path.join('../checkpoints', 'tokenizer')
            self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
            print('loaded tokenizer from', tokenizer_path)

            ## resume selected turn ids
            # load selected turn ids, if have
            self.train_labelled_turn_id_list, self.resumed_round = self.load_csv()
            print(f'length of selected_turn_ids: {len(self.train_labelled_turn_id_list)} resumed_round: {self.resumed_round}')
            # split dialogue id and turn id
            self.train_labelled_dialogue_id_list = [id.split('-')[0] for id in self.train_labelled_turn_id_list]

            # # update the unlabelled pool
            # self.train_unlabelled_dialogue_id_list = np.setdiff1d(self.train_unlabelled_dialogue_id_list, self.train_labelled_dialogue_id_list)
            
        else:
            # Init tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_config.base_model)  # NOTE: by default, unk_token sets to <|endoftext|>
            # Add special tokens
            self.SPECIAL_TOKENS = data_loader.SPECIAL_TOKENS
    #         print(f'Special tokens: {self.SPECIAL_TOKENS}')
            self.tokenizer.add_special_tokens(self.SPECIAL_TOKENS)
            
            
            tokenizer_path = os.path.join(self.config.output_data_dir, 'tokenizer')
            # Save tokenizer for later use
            self.tokenizer.save_pretrained(tokenizer_path)
            print('saved tokenizer to', tokenizer_path)    
            print(f'Tokenizer length: {len(self.tokenizer)}')


        print(f'self.train_labelled_dialogue_id_list: {len(self.train_labelled_dialogue_id_list) if self.train_labelled_dialogue_id_list is not None else 0} {self.train_labelled_dialogue_id_list}')
        print(f'self.train_labelled_turn_id_list: {len(self.train_labelled_turn_id_list) if self.train_labelled_turn_id_list is not None else 0} {self.train_labelled_turn_id_list}')
        print(f'self.train_unlabelled_dialogue_id_list: {len(self.train_unlabelled_dialogue_id_list)} {self.train_unlabelled_dialogue_id_list}')
        
        # Create Metrics Manager
        self.train_metrics = MetricsManager(self.config, data_loader.ds_list, self.value_text2id, self.value_id2text,
                                            self.tokenizer)
        self.valid_metrics = MetricsManager(self.config, data_loader.ds_list, self.value_text2id, self.value_id2text,
                                            self.tokenizer)
        
        self.bos_id, self.eos_id, self.pad_id, self.sep_id = self.tokenizer.convert_tokens_to_ids(
            ['<BOS>', '<EOS>', '<PAD>', '<SEP>'])


    def initialize_model(self):
        """
        Initialize the model with default parameters
        """
        
        print(f'============= Initializing model =============')
        
        # Initialize models
        self.model_config = AutoConfig.from_pretrained(self.config.model_config.base_model)

        from models.KAGE_GPT2.KAGE_GPT2 import KAGEModel

        if self.model is not None:
            del self.model

        self.model = KAGEModel.from_pretrained(self.config.model_config.base_model,
                                                    config=self.model_config,
                                                    sys_config=self.config)  # GPT2LMHeadModel

        self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.to(self.config.device)

        # self.num_labels = self.model.num_labels
        if self.config.freeze_transformer:
            for name, param in self.model.named_parameters():
                if 'transformer' in name:
                    param.requires_grad = False
                print(name, param.requires_grad)

        graph_parameters = filter(lambda p: p[1].requires_grad and 'graph' in p[0], self.model.named_parameters())
        transformer_parameters = filter(lambda p: p[1].requires_grad and 'graph' not in p[0], self.model.named_parameters())

        # Create optimizer
        self.optimizer = optim.AdamW(
            [
                dict(params=[param for name, param in graph_parameters],
                lr=self.config.train.graph_lr,
                eps=self.config.train.adam_epsilon),
                dict(params=[param for name, param in transformer_parameters],
                lr=self.config.train.lr,
                eps=self.config.train.adam_epsilon),
            ]
        )
        
        if self.config.model_config.graph_mode != 'none':
            # Add KB data into model
            value_id2tokenized_text = {}
            ontology_value_id2tokenized_text = {}
            for str_ds_pair in self.ds_list:
                value_id2tokenized_text[str_ds_pair] = {}
                value_dict = self.value_id2text[str_ds_pair]
                for i in range(len(value_dict)):
                    text = value_dict[i]
                    assert text != ''
                    value_id2tokenized_text[str_ds_pair][i] = self.tokenizer(text)['input_ids']
                    # print(text, self.tokenizer(text)['input_ids'])
            self.value_id2tokenized_text = value_id2tokenized_text
            
            for value in self.ontology_value_list:
                assert value != ''
                ontology_value_id2tokenized_text[self.ontology_value_text2id[value]] = self.tokenizer(value)['input_ids']
            self.model.add_KB(
                value_id2tokenized_text,
                self.value_id2text,
                self.ds_list,
                self.ontology_value_list,
                self.ontology_value_text2id,
                self.ontology_value_id2text,
                ontology_value_id2tokenized_text,
            )
        
            
    def active_learning(self):
        """
        For each dialogue, select a turn index to label, which will be added in the training set to train the model.
        For the first iteration, we does not train the model, use the random initialized model to predict the slot value directly
        
        
        in the unlabelled pool, pass a dialogue and all its turns to active_learning()
        """
        
        # resume from previous, we use the pretrained model
        if self.config.continue_select_turn == 1:

            # Initialize models
            from models.KAGE_GPT2.KAGE_GPT2 import KAGEModel

            self.model_config = AutoConfig.from_pretrained('gpt2')
            self.model = KAGEModel.from_pretrained('gpt2',
                                                        config=self.model_config,
                                                        sys_config=self.config)  # GPT2LMHeadModel
            self.model.resize_token_embeddings(len(self.tokenizer))

            self.model.to(self.config.device)

            # Load checkpoints - dont need to consider Sagemaker ENV becasue we run testing in local
            self.sagemaker_load_checkpoint_model(load_model_path=self.load_model_path)

            if self.config.model_config.graph_mode != 'none':
                # Add KB data into model
                value_id2tokenized_text = {}
                ontology_value_id2tokenized_text = {}
                for str_ds_pair in self.ds_list:
                    value_id2tokenized_text[str_ds_pair] = {}
                    value_dict = self.value_id2text[str_ds_pair]
                    for i in range(len(value_dict)):
                        text = value_dict[i]
                        assert text != ''
                        value_id2tokenized_text[str_ds_pair][i] = self.tokenizer(text)['input_ids']
                        # print(text, self.tokenizer(text)['input_ids'])
                self.value_id2tokenized_text = value_id2tokenized_text
                
                for value in self.ontology_value_list:
                    assert value != ''
                    ontology_value_id2tokenized_text[self.ontology_value_text2id[value]] = self.tokenizer(value)['input_ids']
                self.model.add_KB(
                    value_id2tokenized_text,
                    self.value_id2text,
                    self.ds_list,
                    self.ontology_value_list,
                    self.ontology_value_text2id,
                    self.ontology_value_id2text,
                    ontology_value_id2tokenized_text,
                )
        else:
            if self.config.budget <= 0:
                self.initialize_model()
                print('No budget avaiable, thus no inital train model, so we initialize a new model')
        
#         print(f'self.train_unlabelled_dataset: {len(self.train_unlabelled_dataset)}')
#         print(f'self.dialogue_id_list: {len(self.train_unlabelled_dialogue_id_list)}')
#         print(f'self.train_data_loader: {len(self.train_data_loader)}')
#         print(f'self.train_unlabelled_data_loader: {len(self.train_unlabelled_data_loader)}')
#         print(f'self.dialogue_id_list: {self.dialogue_id_list}')

        # split the total unlabelled dialogues into chunks by round
        self.total_round = len(self.train_unlabelled_dialogue_id_list) // self.config.train.num_dialogue_per_round 
        print(f'self.total_round: {self.total_round}')
        
        if self.config.continue_select_turn == 1:
            # update the unlabelled pool
            self.train_unlabelled_dialogue_id_list = np.setdiff1d(self.train_unlabelled_dialogue_id_list, self.train_labelled_dialogue_id_list)

        self.train_unlabelled_dialogue_id_list = np.array(self.train_unlabelled_dialogue_id_list)
        print(f'self.train_unlabelled_dialogue_id_list: {len(self.train_unlabelled_dialogue_id_list)}')
        
        
        start_time = time.time()
        self.select_turn_per_round_time_list = []
        self.total_selected_turn_ids = []
        self.total_selected_dialogue_ids = []
        unselected_dialogue_ids = np.copy(self.train_unlabelled_dialogue_id_list)
        
        
        print('=================== Run Active Learning ===================')
        for round_idx in range(self.total_round+1):
            if self.config.continue_select_turn == 1:
                round_idx = round_idx + self.resumed_round + 1

            per_round_start_time = time.time()
            print(f'============= round index: [{round_idx}/{self.total_round}] =============')
#             print(f'unselected_ids: {len(unselected_dialogue_ids)} {unselected_dialogue_ids}')
            
            if len(unselected_dialogue_ids) == 0:
                continue
            
            if len(unselected_dialogue_ids) < self.config.train.num_dialogue_per_round:
                selected_dialogue_ids = unselected_dialogue_ids
            else:
                selected_dialogue_ids = np.random.choice(unselected_dialogue_ids, self.config.train.num_dialogue_per_round, replace=False)
#             print(f'selected_ids: {len(selected_dialogue_ids)} {selected_dialogue_ids}')
            # update the unselected dialogues
            unselected_dialogue_ids = np.setdiff1d(unselected_dialogue_ids, selected_dialogue_ids)
#             print(f'unselected_ids: {len(unselected_dialogue_ids)} {unselected_dialogue_ids}')
            
           
            self.selected_turn_ids_per_round = []
            self.max_entropy_per_round = []
            self.min_confidence_per_round = []
            # for each dialogue, get the entropy of each turn
            for idx, selected_dialogue_id in enumerate(selected_dialogue_ids):
                print(f'+++++++++++++++ dialogue per round: [{idx}/{len(selected_dialogue_ids)-1}] total round: [{round_idx}/{self.total_round}] current selected_dialogue_id: {selected_dialogue_id} +++++++++++++++')
                self.unlabelled_dataloader, _, = self.data_loader.set_dataloader(self.config, 
                                                                                self.tokenizer, 
                                                                                'train_select_dialogue',
                                                                                'generation',
                                                                                self.value_id2text,
                                                                                self.value_text2id,
                                                                                self.ds_list,
                                                                                self.ds_text2id,
                                                                                data_size=self.load_num,
                                                                                selected_dialogue_ids=[selected_dialogue_id]
                                                                                )
                selected_turn_id, max_entropy, min_confidence = self.select_turn_idx()
                selected_turn_id = f'{selected_dialogue_id}-{selected_turn_id}'
                print(f'selected turn: {selected_turn_id} max_entropy: {max_entropy} min_confidence: {min_confidence}')
                self.selected_turn_ids_per_round.append(selected_turn_id)
                self.max_entropy_per_round.append(max_entropy)
                self.min_confidence_per_round.append(min_confidence)
            
            self.selected_turn_ids_per_round = np.array(self.selected_turn_ids_per_round)
            self.total_selected_turn_ids.extend(self.selected_turn_ids_per_round)
            self.total_selected_dialogue_ids.extend(selected_dialogue_ids)
            print()
            print(f'After round [{round_idx}/{self.total_round}], selected turn ids are: {self.selected_turn_ids_per_round}')

#             print(f'selected_dialogue_ids: {selected_dialogue_ids}')
            print(f'total selected turn ids are budget turn ids (if has) + selected turn ids: len={len(self.total_selected_turn_ids) + len(self.train_labelled_turn_id_list) if self.train_labelled_turn_id_list is not None else 0}')
            
            # skip the training process when reaching the last round
#             if round_idx < self.total_round:
            print('After a round, train the intermediate model using all selected turns...')
            ## now we have selected a turn for a dialogue, we use these data to train the model
            ## currently we use turns per round to trian the model
            # default train the intermediate model using 1 epoch 
            self.train(data_type='train_select_turn', 
                       num_epoch=self.config.round_epoch,
                       selected_dialogue_ids=list(self.train_labelled_dialogue_id_list)+list(self.total_selected_dialogue_ids), ## use accumulated turns from previous rounds
                       selected_turn_ids=list(self.train_labelled_turn_id_list)+list(self.total_selected_turn_ids), ## use accumulated turns from previous rounds
    #                        selected_dialogue_ids=selected_dialogue_ids, ## use only previous round turns
    #                        selected_turn_ids=self.selected_turn_ids_per_round, ## use only previous round turns
#                            run_validation=False
                       run_validation=True, # run generation on Test to check the acc after each round
                       round_idx=round_idx
                      )
                
            select_turn_per_round_time = time.time() - per_round_start_time
            self.select_turn_per_round_time_list.append(select_turn_per_round_time)
            print(f'========> The time needs to select turns per round: {select_turn_per_round_time}')
            
            # save selected ids
            df = pd.DataFrame({
                'round': [round_idx] * len(self.selected_turn_ids_per_round),
                'selected_turn_id': self.selected_turn_ids_per_round,
                'max_entropy': self.max_entropy_per_round,
                'min_confidence': self.min_confidence_per_round,
                'select_turn_per_round_time': [select_turn_per_round_time] * len(self.selected_turn_ids_per_round)
            })
            self.save_csv(df, 'selected_turn_id.csv')
            print()
            
        # after all rounds, we have went through the entire dialogues, and we have selected a turn to label in each dialogue
        print()
        print(f'After all rounds, the total selected turn ids are: len: {len(self.total_selected_turn_ids)} are: {self.total_selected_turn_ids}')
        print(f'Also the initial budget turn ids, len: {len(self.train_labelled_turn_id_list)} are: {self.train_labelled_turn_id_list}')
        
        # end time of select turns
        select_turn_end_time = time.time() - start_time
        
        print(f'The time needs to select all turns all rounds: {select_turn_end_time}')
        print()

#         print('All turns are selected, start to train the model...')
#         # now we use only selected turns to train the model and save checkpoints
#         self.train(data_type='train', 
#                    epoch=int(self.config.train.epochs),
#                    selected_dialogue_ids=None, 
#                    selected_turn_ids=self.total_selected_turn_ids, 
#                    run_validation=True
#                   )
        
                

        
        
            
    def select_turn_idx(self):
        """
        For each dialogue, select a turn index to label, which will be added in the training set to train the model.
        For the first iteration, we does not train the model, use the random initialized model to predict the slot value directly
        
        1. , given a dialogue, predict each slot value a

        return: selected_turn_id, max_entropy(if has, otherwise is 0), least_acc(if has, otherwise is 0)
        """

        # select turn on the acquisition method
        if self.config.acquisition == 'random':
            print(f'use Random to select turn')
#             print(f'len train_unlabelled_data_loader: {len(self.unlabelled_dataloader)}')

            assert len(self.unlabelled_dataloader) == 1
            for i_batch, sample_batched in enumerate(self.unlabelled_dataloader):
                input_ids = sample_batched['input_ids']
                
                selected_turn_id = random.randint(0, len(input_ids) - 1)
#                 print(len(input_ids))
#                 print(f'selected id: {selected_turn_id}')

            return selected_turn_id, 0, 0
              
        elif self.config.acquisition == 'last_turn':
            print(f'use Last Turn to select turn')
#             print(f'len train_unlabelled_data_loader: {len(self.unlabelled_dataloader)}')

            assert len(self.unlabelled_dataloader) == 1
            for i_batch, sample_batched in enumerate(self.unlabelled_dataloader):
                input_ids = sample_batched['input_ids']
                
                selected_turn_id = len(input_ids) - 1
#                 print(len(input_ids))
#                 print(f'selected id: {selected_turn_id}')

            return selected_turn_id, 0, 0

        elif self.config.acquisition == 'least_confidence':
            print(f'use Least Confidence to select turn')
            print('=================== Run prediction on Unlabelled dataset ===================')
            
            with torch.no_grad():
                self.model.eval()
                # self.model.set_module_to_eval()
                self.valid_metrics.init_session()

                if self.config.model_config.graph_mode != 'none':
                    # refresh value node embeddings
                    self.model.refresh_embeddings()
                    
                confidence_list = []
                print(f'len train_unlabelled_data_loader: {len(self.unlabelled_dataloader)}')
                for i_batch, sample_batched in enumerate(self.unlabelled_dataloader):

                    input_ids = sample_batched['input_ids'].to(self.config.device)
                    attention_mask = sample_batched['attention_mask'].to(self.config.device)
                    token_type_ids = sample_batched['token_type_ids']
                    if token_type_ids:
                        token_type_ids = token_type_ids.to(self.config.device)
                    pre_input_ids = sample_batched['pre_input_ids'].to(self.config.device)
                    pre_attention_mask = sample_batched['pre_attention_mask'].to(self.config.device)
                    pre_ds_indice = sample_batched['pre_ds_indice']
                    ds_ids = sample_batched['ds_ids']
                    batch_size, ctx_len = input_ids.size()
    #                 assert batch_size == 1
                    max_len = min(ctx_len + 300, 1024)
                    
    #                 print(f'the last token of input_ids: {input_ids[:, -1]}')
                    output, _, max_predicted_val_list = self.model.generate(
                                                    input_ids,
                                                    max_length=max_len,
                                                    do_sample=False,
                                                    temperature=1.0, use_cache=True,
                                                    num_beams=1,
                                                    pre_input_ids=pre_input_ids,
                                                    pre_attention_mask=pre_attention_mask,
                                                    pre_ds_indice=pre_ds_indice,
                                                    bos_id=self.bos_id,
                                                    eos_token_id=self.eos_id,
                                                    pad_token_id=self.pad_id,
                                                    sep_token_id=self.sep_id,
                                                    ds_ids=ds_ids,
                                                    early_stopping=True)

                    # ootput_ids: B * x
                    ootput_ids = output.cpu().numpy().tolist()
                    
                    # self.valid_metrics.add_turn_results_gen_test(ootput_ids, sample_batched)
                    max_predicted_val_list = max_predicted_val_list.cpu().numpy().tolist()
                    confidence_list.extend(max_predicted_val_list)

            print(f'confidence_list: {confidence_list}')
            confidence_list = torch.Tensor(confidence_list)
            selected_turn_id = torch.argmin(confidence_list).item()
            min_confidence = torch.min(confidence_list).item()
            
            # predict_slot_acc_list = self.valid_metrics.turn_slot_acc_list
            # print(f'predict_slot_acc_list: {predict_slot_acc_list}')
            # predict_slot_acc_list = np.array(predict_slot_acc_list)
            # selected_turn_id = np.argmin(predict_slot_acc_list)
            # min_slot_acc = np.amin(predict_slot_acc_list)
            
            return selected_turn_id, 0, min_confidence
            
        elif self.config.acquisition == 'max_entropy':
            print(f'use Max Entropy to select turn')
            print('=================== Run prediction on Unlabelled dataset ===================')
            
            with torch.no_grad():
                self.model.eval()
                # self.model.set_module_to_eval()
                self.valid_metrics.init_session()

                if self.config.model_config.graph_mode != 'none':
                    # refresh value node embeddings
                    self.model.refresh_embeddings()
                    
                entropy_list = []
                print(f'len train_unlabelled_data_loader: {len(self.unlabelled_dataloader)}')
                for i_batch, sample_batched in enumerate(self.unlabelled_dataloader):

                    input_ids = sample_batched['input_ids'].to(self.config.device)
                    attention_mask = sample_batched['attention_mask'].to(self.config.device)
                    token_type_ids = sample_batched['token_type_ids']
                    if token_type_ids:
                        token_type_ids = token_type_ids.to(self.config.device)
                    pre_input_ids = sample_batched['pre_input_ids'].to(self.config.device)
                    pre_attention_mask = sample_batched['pre_attention_mask'].to(self.config.device)
                    pre_ds_indice = sample_batched['pre_ds_indice']
                    ds_ids = sample_batched['ds_ids']
                    batch_size, ctx_len = input_ids.size()
    #                 assert batch_size == 1
                    max_len = min(ctx_len + 300, 1024)
                    
    #                 print(f'the last token of input_ids: {input_ids[:, -1]}')
                    output, generated_output_entropy, _ = self.model.generate(
                                                    input_ids,
                                                    max_length=max_len,
                                                    do_sample=False,
                                                    temperature=1.0, use_cache=True,
                                                    num_beams=1,
                                                    pre_input_ids=pre_input_ids,
                                                    pre_attention_mask=pre_attention_mask,
                                                    pre_ds_indice=pre_ds_indice,
                                                    bos_id=self.bos_id,
                                                    eos_token_id=self.eos_id,
                                                    pad_token_id=self.pad_id,
                                                    sep_token_id=self.sep_id,
                                                    ds_ids=ds_ids,
                                                    early_stopping=True)

                    # ootput_ids: B * x
                    ootput_ids = output.cpu().numpy().tolist()
                    
    #                 self.valid_metrics.add_turn_results_gen_test(ootput_ids, sample_batched)
                    generated_output_entropy = generated_output_entropy.cpu().numpy().tolist()
                    entropy_list.extend(generated_output_entropy)

            print(f'entropy_list: {entropy_list}')
            entropy_list = torch.Tensor(entropy_list)
            selected_turn_id = torch.argmax(entropy_list).item()
            max_entropy = torch.max(entropy_list).item()
            
            return selected_turn_id, max_entropy, 0
    
    
    def budget_train(self, num_epoch):
        """
        Use budget to train an initial model
        """
        
        # train for 3 epochs
        self.train(
            data_type='train', 
            num_epoch=num_epoch,
            selected_dialogue_ids=self.train_labelled_dialogue_id_list, ## use accumulated turns from previous rounds
            selected_turn_ids=self.train_labelled_turn_id_list, ## use accumulated turns from previous rounds
            run_validation=False
        )
        
        

    def train(self, data_type='train', num_epoch=1, selected_dialogue_ids=None, selected_turn_ids=None, run_validation=False, round_idx=0):
        #############################################
        #
        #                load setup
        #
        #############################################
        
        # initialize the model
        self.initialize_model()
        
        # load training daata
        (
            self.train_data_loader, 
            train_dataset
        ) = self.data_loader.set_dataloader(self.config, self.tokenizer, data_type,
                                                                           'teacher_force',
                                                                           self.value_id2text,
                                                                           self.value_text2id,
                                                                           self.ds_list,
                                                                           self.ds_text2id,
                                                                           data_size=self.load_num,
                                                                           selected_dialogue_ids=selected_dialogue_ids,
                                                                           selected_turn_ids=selected_turn_ids
                                           )
        print(f'len train data loader: {len(self.train_data_loader.dataset)}')
#         self.valid_data_loader, valid_dataset = self.data_loader.set_dataloader(config, self.tokenizer, 'dev',
#                                                                            'teacher_force',
#                                                                            self.value_id2text,
#                                                                            self.value_text2id,
#                                                                            self.ds_list,
#                                                                            self.ds_text2id,
#                                                                            data_size=load_num)
        if run_validation:
            print('Using [VALID] set to run generation validation')
            (
                self.valid_gen_data_loader, 
                _
            ) = self.data_loader.set_dataloader(self.config, 
                                                self.tokenizer, 
                                                'dev',
                                                'generation',
                                                self.value_id2text,
                                                self.value_text2id,
                                                self.ds_list,
                                                self.ds_text2id,
                                                data_size=self.load_num)
            
            print(f'[VALID] set length: {len(self.valid_gen_data_loader)}')
            print('Using [TEST] set to test acc')
            (
                self.test_data_loader, 
                _
            ) = self.data_loader.set_dataloader(self.config, 
                                                self.tokenizer, 
                                                'test',
                                                'generation',
                                                self.value_id2text,
                                                self.value_text2id,
                                                self.ds_list,
                                                self.ds_text2id,
                                                data_size=self.load_num)
            print(f'[TEST] set length: {len(self.test_data_loader)}')


#         t_total = len(
#             self.train_data_loader) // self.config.train.additional.gradient_accumulation_steps * (
#                               self.config.train.epochs)
        t_total = len(self.train_data_loader) // self.config.train.additional.gradient_accumulation_steps * (num_epoch)
        print(f'===============> t_total iterations: {t_total}')
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.train.additional.warmup_steps,
            num_training_steps=t_total
        )

#         # Load checkpoints
#         if self.config.ENV == 'sagemaker_remote':
#             # running on Sagemaker training job
#             if self.config.use_spot_instances:
#                 # if use Spot training, save the checkpoint to opt/ml/checkpoint
#                 # set to <= 1 because we have already saved tokenzier/ folder
#                 if len(os.listdir(self.config.checkpoint_path)) <= 1:
#                     # check if has previous checkpoint or not
#                     print('checkpoints do not exists! Set epoch to 0')
#                     self.loaded_epoch = 0
#                 else:
#                     # get the latest epoch
#                     print('checkpoint folder exists!')
#                     print(os.listdir(self.config.checkpoint_path))
#                     checkpoints = [f for f in os.listdir(self.config.checkpoint_path) if os.path.isfile(os.path.join(self.config.checkpoint_path, f))]

#                     latest_checkpoint_filename = max(checkpoints, key=extract_number_in_filename)
#                     print(latest_checkpoint_filename)

#                     self.sagemaker_load_checkpoint_model(latest_checkpoint_filename, load_epoch=self.config.train.load_epoch,
#                                            load_best_model=self.config.train.load_best_model,
#                                            load_model_path=self.config.train.load_model_path)
#             else:
#                 # if not using spot training, then start training from the begining
#                 print('Not using Spot Training! Set epoch to 0 to start the training from the begining!')
#                 self.loaded_epoch = 0
#         else:
#             # running on local
#             print('Set epoch to 0 to start the training from the begining!')
#             self.loaded_epoch = 0
# #             self.load_checkpoint_model(load_epoch=self.config.train.load_epoch,
# #                                        load_best_model=self.config.train.load_best_model,
# #                                        load_model_path=self.config.train.load_model_path)

        print('Set epoch to 0 to start the training from the begining!')
        self.loaded_epoch = 0

#         print("finished initialization...starting training.")
        
#         # Create Metrics Manager
#         self.train_metrics = MetricsManager(self.config, data_loader.ds_list, self.value_text2id, self.value_id2text,
#                                             self.tokenizer)
#         self.valid_metrics = MetricsManager(self.config, data_loader.ds_list, self.value_text2id, self.value_id2text,
#                                             self.tokenizer)

        
        

#         batch_size = self.config.train.batch_size
        save_interval = self.config.train.save_interval
        device = self.config.device
        gradient_accumulation_steps = self.config.train.additional.gradient_accumulation_steps
        start_time = datetime.datetime.now()
        
        if self.config.ENV == 'sagemaker_remote':
            # running on Sagemaker training job
            logdir = self.config.sm_tensorboard_path  
        else:
            # running on local
            logdir = os.path.join(self.config.tensorboard_path)  
#         print(f'tensorboard save dir: {logdir}')

        if self.config.fp16:
            # Creates once at the beginning of training
            scaler = torch.cuda.amp.GradScaler()

        ADDED_GRAPH = True
        if self.config.reset:
            ADDED_GRAPH = False

        if self.loaded_epoch == 0:
            ADDED_GRAPH = False

        writer = SummaryWriter(logdir)

        # bos_id, eos_id, pad_id, sep_id = self.tokenizer.convert_tokens_to_ids(
        #     ['<BOS>', '<EOS>', '<PAD>', '<SEP>'])
        
        print('============================= Training starts! =============================')

        # early stop criteria
        patience = 2
        best_score = {'epoch': 0, 'joint_acc': 0, 'slot_acc': 0}
        avg_train_loss_list = []
        for epoch in range(num_epoch):
            current_epoch = epoch + self.loaded_epoch + 1
#             print(f'Current epoch: {current_epoch} self.loaded_epoch: {self.loaded_epoch}')
#             if current_epoch > int(self.config.train.epochs):
#                 print('Training completed.')
#                 break
            #############################################
            #
            #                Train
            #
            #############################################
            # zero the parameter gradients
            self.model.train()
            # self.model.set_module_to_train()
            total_loss_list = []
            cls_loss_list = []
            gen_loss_list = []
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                if i_batch == 0:
                    self.optimizer.zero_grad()

                # At each batch
                input_ids = sample_batched['input_ids'].to(self.config.device)
                attention_mask = sample_batched['attention_mask'].to(self.config.device)
                token_type_ids = sample_batched['token_type_ids']
                if token_type_ids:
                    token_type_ids = token_type_ids.to(self.config.device)
                pre_input_ids = sample_batched['pre_input_ids'].to(self.config.device)
                pre_attention_mask = sample_batched['pre_attention_mask'].to(self.config.device)
                pre_ignore_len = sample_batched['pre_ignore_len']
                pre_ds_indice = sample_batched['pre_ds_indice']
                ds_indice = sample_batched['ds_indice']
                
                cls_labels = sample_batched['cls_labels'] # B x 30
                gen_labels = sample_batched['label_ids'].to(self.config.device)
                cls_labels = torch.LongTensor(cls_labels).to(self.config.device)#.int()
                
#                 cls_label_dict = sample_batched['cls_label_dict'] # B x 30
#                 example_id = sample_batched['example_id']
#                 context = sample_batched['context']
#                 turn_utt = sample_batched['turn_utt']

                if self.config.fp16:
                    # Casts operations to mixed precision
                    with torch.cuda.amp.autocast():
                        pre_forward_results = self.model.pre_forward(
                                                pre_input_ids=pre_input_ids,
                                                pre_attention_mask=pre_attention_mask,
                                                pre_ignore_len=None,
                                                pre_ds_indice=pre_ds_indice,
                                            )
                        ds_embeddings = pre_forward_results['ds_embeddings']
                        graph_forward_results = self.model.graph_forward(
                            ds_embeddings=ds_embeddings
                        )
                        ds_embeddings = graph_forward_results['ds_embeddings']

                        forward_results = self.model(input_ids=input_ids,
                                                     attention_mask=attention_mask,
                                                     token_type_ids=token_type_ids,
                                                     labels=gen_labels,
                                                     # list_cls_index=list_cls_index,
                                                     return_dict=True,
                                                     ds_indice=ds_indice,
                                                     ds_embeddings=ds_embeddings,
                                                     )
                else:
                    pre_forward_results = self.model.pre_forward(
                        pre_input_ids=pre_input_ids,
                        pre_attention_mask=pre_attention_mask,
                        pre_ignore_len=None,
                        pre_ds_indice=pre_ds_indice,
                    )
                    ds_embeddings = pre_forward_results['ds_embeddings']
                
                    graph_forward_results = self.model.graph_forward(
                        ds_embeddings=ds_embeddings,
                        cls_labels=cls_labels,
                    )
                    ds_embeddings = graph_forward_results['ds_embeddings']
                    cls_loss = graph_forward_results['loss']

                    forward_results = self.model(input_ids=input_ids,
                                                 attention_mask=attention_mask,
                                                 token_type_ids=token_type_ids,
                                                 labels=gen_labels,
                                                 # list_cls_index=list_cls_index,
                                                 return_dict=True,
                                                 ds_indice=ds_indice,
                                                 ds_embeddings=ds_embeddings,
                                                 )
                
                logits = forward_results['logits']

                # cls_logits = forward_results['cls_logits']
                # cls_loss = self.compute_cls_loss(cls_logits, cls_labels, list_cls_index)
                # gen_loss = self.compute_gen_loss(logits, gen_labels)
                gen_loss = forward_results['loss']
                total_loss = gen_loss + cls_loss * 0.001
                total_loss = total_loss / gradient_accumulation_steps
                
                if i_batch % 100 == 0:
                    print(f"[TRAIN] epoch {current_epoch} - batch {i_batch} - train loss {total_loss}=({gen_loss} + {cls_loss}) - {i_batch}/{len(self.train_data_loader)}")
                
                
                if self.config.fp16:
                    # Scales the loss, and calls backward()
                    # to create scaled gradients
                    scaler.scale(total_loss).backward()
                else:
                    total_loss.backward(retain_graph=False)

                # self.optimizer.step()
                if i_batch % gradient_accumulation_steps == 0 and i_batch != 0:

                    if self.config.fp16:
                        # Unscales the gradients of optimizer's assigned params in-place
                        scaler.unscale_(self.optimizer)
                        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       self.config.train.additional.gradient_clipping)
                        # Unscales gradients and calls
                        # or skips optimizer.step()
                        scaler.step(self.optimizer)
                        # Updates the scale for next iteration
                        scaler.update()
                    else:
                        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                              self.config.train.additional.gradient_clipping)
#                         print('optimizer step!')
                        self.optimizer.step()

                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()
#                         print('scheduler step! LR:', self.scheduler.get_last_lr())
#                         print([group['lr'] for group in self.optimizer.param_groups])

                    if self.config.model_config.graph_mode != 'none':
                        # refresh value node embeddings after back propagation
                        self.model.refresh_embeddings()


                gen_predictions = torch.argmax(logits.detach().cpu(), dim=-1)

                cls_logits = graph_forward_results['logits']
                cls_predictions = []
                for cls_logit in cls_logits:
                    cls_predictions.append(torch.argmax(cls_logit, dim=-1).detach().cpu())

                if i_batch % 50 == 0 and i_batch != 0:
                    self.train_metrics.add_turn_results_cls(None, cls_predictions, None, sample_batched, verbose=True)
                    self.train_metrics.add_turn_results_gen(gen_predictions, sample_batched, verbose=True)
                else:
                    self.train_metrics.add_turn_results_cls(None, cls_predictions, None, sample_batched)
                    self.train_metrics.add_turn_results_gen(gen_predictions, sample_batched)

                total_loss_list.append(total_loss.detach().cpu().numpy())
                gen_loss_list.append(gen_loss.detach().cpu().numpy())
                if self.config.model_config.cls_loss:
                    cls_loss_list.append(cls_loss.detach().cpu().numpy())

            self.optimizer.step()
            self.optimizer.zero_grad()

            ### train loss after each epoch NEED TO SAVE THEM IN A CSV
            current_epoch_train_loss = np.mean(np.array(total_loss_list))
            current_epoch_gen_loss = np.mean(np.array(gen_loss_list))
            avg_train_loss_list.append(current_epoch_train_loss)
            # avg_train_total_loss_per_epoch = np.mean(np.array(total_loss_list))
            print(f"epoch {current_epoch} - avg train loss {current_epoch_train_loss}")

            train_loss_df = pd.DataFrame({
                'round': [round_idx],
                'epoch': [current_epoch],
                'avg_train_loss': [current_epoch_train_loss],
            })
            self.save_csv(train_loss_df, 'train_loss_per_epoch_per_round.csv')
            # print('\n')

            # if self.config.model_config.cls_loss:
            #     writer.add_scalar('train/cls_loss', np.mean(np.array(cls_loss_list)), current_epoch)
            

            
            ############################################################
            ############### valid on Validation set
            ############### determine when to stop training
            ############################################################
            if run_validation:
                # only skip validation in previous rounds, in the final round, we evaluate for each epoch
                if round_idx != self.total_round:
                    # Skip validation with valid.set_size
                    if self.config.valid.step_size > 1:
                        # for sparse training: self.config.valid.step_size=3, 当epoch不等于3,6,9...时，跳过；也就是当epoch为3的倍数时，运行下面的validation 
                        if current_epoch % self.config.valid.step_size != 0:
                            print('skip validation...')
                            continue
                
                # in the final round, skip validation for the first 5 epochs
                if round_idx == self.total_round and current_epoch <= 7:
                    print('skip validation for early epochs')
                    continue
                elif round_idx == 0 and current_epoch <= 23:
                    # start valid from 24
                    print(f'at round-{round_idx}, skip validation for early epochs')
                    continue
                elif round_idx == 1 and current_epoch <= 14:
                    print(f'at round-{round_idx}, skip validation for early epochs')
                    continue
                elif round_idx == 2 and current_epoch <= 7:
                    print(f'at round-{round_idx}, skip validation for early epochs')
                    continue

                print('=================== Run generation validation on [VALID] set ===================')
                with torch.no_grad():
                    self.model.eval()
                    # self.model.set_module_to_eval()
                    self.valid_metrics.init_session()

                    if self.config.model_config.graph_mode != 'none':
                        # refresh value node embeddings
                        self.model.refresh_embeddings()

    #                 entropy_list = []
                    for i_batch, sample_batched in enumerate(self.valid_gen_data_loader):
                        # if i_batch >= self.config.valid.num_valid_generation:
                        #     break
                        input_ids = sample_batched['input_ids'].to(self.config.device)
                        attention_mask = sample_batched['attention_mask'].to(self.config.device)
                        token_type_ids = sample_batched['token_type_ids']
                        if token_type_ids:
                            token_type_ids = token_type_ids.to(self.config.device)
                        pre_input_ids = sample_batched['pre_input_ids'].to(self.config.device)
                        pre_attention_mask = sample_batched['pre_attention_mask'].to(self.config.device)
                        pre_ds_indice = sample_batched['pre_ds_indice']
                        ds_ids = sample_batched['ds_ids']
                        batch_size, ctx_len = input_ids.size()
#                         assert batch_size == 1
                        max_len = min(ctx_len + 300, 1024)

                        output, _, _ = self.model.generate(input_ids,
                                                     max_length=max_len,
                                                     do_sample=False,
                                                     temperature=1.0, use_cache=True,
                                                     num_beams=1,
                                                     pre_input_ids=pre_input_ids,
                                                     pre_attention_mask=pre_attention_mask,
                                                     pre_ds_indice=pre_ds_indice,
                                                     bos_id=self.bos_id,
                                                     eos_token_id=self.eos_id,
                                                     pad_token_id=self.pad_id,
                                                     sep_token_id=self.sep_id,
                                                     ds_ids=ds_ids,
                                                     early_stopping=True)

                        ootput_ids = output.cpu().numpy().tolist()
                        self.valid_metrics.add_turn_results_gen_test(ootput_ids, sample_batched)

                    # After validation, print results
                    metrics = self.valid_metrics.get_metrics()
                    # print(metrics)

                    # UPDATE: print validation generation scores
#                     if (metrics['general']['joint_acc'] > best_score['joint_acc']) or (metrics['general']['slot_acc'] >  best_score['slot_acc']):
                    if (metrics['general']['joint_acc'] > best_score['joint_acc']):
                        best_score['epoch'] = current_epoch
                        best_score['joint_acc'] = metrics['general']['joint_acc']
                        best_score['slot_acc'] = metrics['general']['slot_acc']

                        # reset, because we want 3 continuous epochs to not increase the acc 
                        patience = 2

                        # print(f'current round: {round_idx} total_round: {self.total_round}')
                        # if round_idx == self.total_round:
                            # only save model when its the final round

                        # save model when current epoch performance is greater
                        if self.config.ENV == 'sagemaker_remote':
                            # Running on Sagemaker training job
                            if self.config.use_spot_instances:
                                # if use Spot training, save the checkpoint to opt/ml/checkpoint
                                self.save_model_prefix = self.config.checkpoint_path
                                self.sagemaker_save_checkpoint(self.save_model_prefix, current_epoch_train_loss, current_epoch_gen_loss, current_epoch, round_idx)
                            else:
                                # no using Spot training, save the checkpoint to /opt/ml/output/non_spot_checkpoints
                #                 self.save_model_prefix = self.config.non_spot_checkpoint_path
                                self.save_model_prefix = self.config.output_data_dir
                                self.sagemaker_save_checkpoint(self.save_model_prefix, current_epoch_train_loss, current_epoch_gen_loss, current_epoch, round_idx)
                        else:
                            # running on local
                            # self.save_checkpoint('{}'.format(current_epoch), record_best_model=False)
                            self.save_model_prefix = self.config.saved_model_path
                            self.local_save_checkpoint(self.save_model_prefix, current_epoch, round_idx)
                    
                    else:
                        if metrics['general']['joint_acc'] == 0 and metrics['general']['slot_acc'] == 0:
                            # handle the case when initially joint_acc and slot_acc are 0
                            pass
                        else:
                            # in current epoch, valid acc does not increase
                            patience -= 1
                            print(f'[No improve] at current epoch {current_epoch}, valid acc does not increase')

                    print(f'number of TRAINED examples: {len(self.train_data_loader)} | number of VALID examples: {len(self.valid_gen_data_loader)}')
                    print(f"round: [{round_idx}/{self.total_round}] current epoch: [{current_epoch}/{num_epoch}] valid joint_acc: {metrics['general']['joint_acc']} valid slot_acc: {metrics['general']['slot_acc']} || best epoch: {best_score['epoch']} valid joint_acc: {best_score['joint_acc']} best valid slot_acc: {best_score['slot_acc']}")

                    # since we use skip validation, so if after 1 epochs, not greater than previous one, we stop epoch loop
                    if patience == 0:
                        current_epoch -= patience * 2
                        print(f'******************* [Early stopping] stop at epoch {current_epoch} *******************')
                        # break out of the epoch loop
                        break
                        
                    
                    # save acc on Valid set after each round
                    valid_df = pd.DataFrame({
                        'round': [round_idx],
                        'epoch': [current_epoch],
                        'valid_joint_acc': [metrics['general']['joint_acc']],
                        'valid_slot_acc': [metrics['general']['slot_acc']],
                    })
                    self.save_csv(valid_df, 'valid_acc_per_round.csv')
                    print('\n')

                    self.valid_metrics.init_session()
                    # writer.flush()
                    # print('Generation validation on [VALID] results added!')
        
        # # save acc on Valid set after each round
        # df = pd.DataFrame({
        #     'round': [round_idx],
        #     'best_epoch': [best_score['epoch']],
        #     'valid_joint_acc': [best_score['joint_acc']],
        #     'valid_slot_acc': [best_score['slot_acc']],
        # })
        # self.save_csv(df, 'valid_acc_per_round.csv')
        # print('\n')

        # run inference on TEST set
        self.run_inference(round_idx, current_epoch)


    def run_inference(self, round_idx, current_epoch):
        print('\n')
        print(f'==================== Run inference on [TEST] SET ====================')

        from models.KAGE_GPT2.KAGE_GPT2 import KAGEModel

        if self.model is not None:
            del self.model

        model_config = AutoConfig.from_pretrained('gpt2')

        self.model = KAGEModel.from_pretrained('gpt2',
                                        config=model_config,
                                        sys_config=self.config)  # GPT2LMHeadModel
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.to(self.config.device)

        ## load saved model
        if self.config.ENV == 'sagemaker_remote':
            # Running on Sagemaker training job
            self.save_model_prefix = self.config.output_data_dir
        else:
            # running on local
            self.save_model_prefix = self.config.saved_model_path

        self.load_saved_model(self.save_model_prefix)

        # load saved model
        print("finished loading saved model...starting evaluation.")

        self.test_metrics = MetricsManager(self.config, self.data_loader.ds_list, self.value_text2id, self.value_id2text,
                                            self.tokenizer)


        if self.config.model_config.graph_mode != 'none':
            # Add KB data into model
            value_id2tokenized_text = {}
            ontology_value_id2tokenized_text = {}
            for str_ds_pair in self.ds_list:
                value_id2tokenized_text[str_ds_pair] = {}
                value_dict = self.value_id2text[str_ds_pair]
                # print(str_ds_pair, value_dict)
                for i in range(len(value_dict)):
                    text = value_dict[i]
                    assert text != ''
                    value_id2tokenized_text[str_ds_pair][i] = self.tokenizer(text)['input_ids']
                    # print(text, self.tokenizer(text)['input_ids'])
            self.value_id2tokenized_text = value_id2tokenized_text

            for value in self.ontology_value_list:
                assert value != ''
                ontology_value_id2tokenized_text[self.ontology_value_text2id[value]] = self.tokenizer(value)['input_ids']
            self.model.add_KB(
                value_id2tokenized_text,
                self.value_id2text,
                self.ds_list,
                self.ontology_value_list,
                self.ontology_value_text2id,
                self.ontology_value_id2text,
                ontology_value_id2tokenized_text,
            )


        with torch.no_grad():
            self.model.eval()
            # self.model.set_module_to_eval()
            self.test_metrics.init_session()

            # if self.config.model_config.graph_mode != 'none':
            #     # refresh value node embeddings
            #     self.model.refresh_embeddings()

#                 entropy_list = []
            for i_batch, sample_batched in enumerate(self.test_data_loader):
                
                input_ids = sample_batched['input_ids'].to(self.config.device)
                attention_mask = sample_batched['attention_mask'].to(self.config.device)
                token_type_ids = sample_batched['token_type_ids']
                if token_type_ids:
                    token_type_ids = token_type_ids.to(self.config.device)
                pre_input_ids = sample_batched['pre_input_ids'].to(self.config.device)
                pre_attention_mask = sample_batched['pre_attention_mask'].to(self.config.device)
                pre_ds_indice = sample_batched['pre_ds_indice']
                ds_ids = sample_batched['ds_ids']
                batch_size, ctx_len = input_ids.size()
#                         assert batch_size == 1
                max_len = min(ctx_len + 300, 1024)

                output, _, _ = self.model.generate(input_ids,
                                                max_length=max_len,
                                                do_sample=False,
                                                temperature=1.0, use_cache=True,
                                                num_beams=1,
                                                pre_input_ids=pre_input_ids,
                                                pre_attention_mask=pre_attention_mask,
                                                pre_ds_indice=pre_ds_indice,
                                                bos_id=self.bos_id,
                                                eos_token_id=self.eos_id,
                                                pad_token_id=self.pad_id,
                                                sep_token_id=self.sep_id,
                                                ds_ids=ds_ids,
                                                early_stopping=True)

                ootput_ids = output.cpu().numpy().tolist()
                self.test_metrics.add_turn_results_gen_test(ootput_ids, sample_batched)

            # After validation, print results
            metrics = self.test_metrics.get_metrics()
            # print(metrics)

            # Save metrics
            if self.config.ENV == 'sagemaker_remote':
                # Running on Sagemaker training job
                save_result_prefix = self.config.output_data_dir
            else:
                # running on local
                save_result_prefix = self.config.saved_model_path
                
            if not os.path.exists(save_result_prefix):
                os.mkdir(save_result_prefix)
                
            save_general_result_prefix = f'{save_result_prefix}/metrics_round{round_idx}.json'
            save_turn_result_prefix = f'{save_result_prefix}/test_results_round{round_idx}.json'

            with open(save_general_result_prefix, 'w') as f1:
                json.dump(metrics, f1, indent=4)
                print('metrics has been saved to', save_general_result_prefix)

            with open(save_turn_result_prefix, 'w') as f2:
                json.dump(self.test_metrics.results, f2, indent=4)
                print(f'result has been saved to {save_turn_result_prefix}')

            # # Save to result path
            # self.test_metrics.save_results('{}_results.json'.format('test'))

#             # UPDATE: print validation generation scores
#             if (metrics['general']['joint_acc'] >  best_score['joint_acc']) or (metrics['general']['slot_acc'] >  best_score['slot_acc']):
#                 best_score['epoch'] = current_epoch
#                 best_score['joint_acc'] = metrics['general']['joint_acc']
#                 best_score['slot_acc'] = metrics['general']['slot_acc']
                
# #                         # save acc on Test set after each round
# #                         df = pd.DataFrame({
# #                             'round': [round_idx],
# #                             'epoch': [best_score['epoch']],
# #                             'joint_acc': [best_score['joint_acc']],
# #                             'slot_acc': [best_score['slot_acc']],
# #                         })
# #                         self.save_csv(df, 'acc_per_round.csv')
# #                         print('\n')
                
# #                         # save model
# #                         if self.config.ENV == 'sagemaker_remote':
# #                             # Running on Sagemaker training job
# #                             if self.config.use_spot_instances:
# #                                 # if use Spot training, save the checkpoint to opt/ml/checkpoint
# #                                 self.save_model_prefix = self.config.checkpoint_path
# #                                 self.sagemaker_save_checkpoint(self.save_model_prefix, current_epoch_train_loss, current_epoch_gen_loss, '{}'.format(current_epoch))
# #                             else:
# #                                 # no using Spot training, save the checkpoint to /opt/ml/output/non_spot_checkpoints
# #                 #                 self.save_model_prefix = self.config.non_spot_checkpoint_path
# #                                 self.save_model_prefix = self.config.output_data_dir
# #                                 self.sagemaker_save_checkpoint(self.save_model_prefix, current_epoch_train_loss, current_epoch_gen_loss, '{}'.format(current_epoch))
# #                         else:
# #                             # running on local
# #                             self.save_checkpoint('{}'.format(current_epoch), record_best_model=False)
            print(f'number of TRAINED examples: {len(self.train_data_loader)} | number of TEST examples: {len(self.test_data_loader)}')
            print(f"round: {round_idx} current epoch: {current_epoch} test joint_acc: {metrics['general']['joint_acc']} test slot_acc: {metrics['general']['slot_acc']}")

            # save acc on Test set after each round
            test_df = pd.DataFrame({
                'round': [round_idx],
                'epoch': [current_epoch],
                'test_joint_acc': [metrics['general']['joint_acc']],
                'test_slot_acc': [metrics['general']['slot_acc']],
            })
            self.save_csv(test_df, 'test_acc_per_round.csv')
            print('\n')

#             self.test_metrics.init_session()
#             # writer.flush()
#             print('Generation prediction on [TEST] results added!')