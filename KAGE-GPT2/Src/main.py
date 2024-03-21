"""
The code is adapted from:
main.py: main file for the paper:
Lin, W., Tseng, B. H., & Byrne, B. (2021). Knowledge-Aware Graph-Enhanced GPT-2 for Dialogue State Tracking. EMNLP 2021.
https://arxiv.org/abs/2104.04466v3
"""

__author__ = "Weizhe Lin"
__copyright__ = "Copyright 2021, Weizhe Lin"
__version__ = "1.0.0"
__email__ = "wl356@cam.ac.uk"
__status__ = "Published for Github"

import os
import argparse

from utils.config_system import process_config
from utils.log_system import logger
from utils.dirs import *
from pprint import pprint
import torch
from utils.cuda_stats import print_cuda_statistics
import json
import numpy as np
import random

def main(config):
#     pprint(config)
    if config.model == 'DSTQA':
        from train_executor.DSTQA_executor import DSTQAExecutor
        from test_evaluator.DSTQA_evaluator import DSTQAEvaluator
        Train_Executor = DSTQAExecutor
        Test_Evaluator = DSTQAEvaluator
    elif config.model == 'KAGE':
        from train_executor.KAGE_executor import KAGEExecutor
#         from train_executor.turn_selector import TurnSelector
        from test_evaluator.KAGE_evaluator import KAGEEvaluator
        Train_Executor = KAGEExecutor
#         Turn_Selector = TurnSelector
        Test_Evaluator = KAGEEvaluator
    else:
#         logger.print(config.model, "is not a valid model type! Please check.", mode="warning")
        return

    if config.data_loader.type == 'data_loader_dstqa':
        DataLoaderWrapper = None
    elif config.data_loader.type == 'data_loader_kage':
        from data_loader_manager.data_loader_kage import DataLoaderWrapper
    else:
#         logger.print(config.data_loader.type, "is not a valid data loader type! Please check.", mode="warning")
        return

    if config.mode == 'train':
        if DataLoaderWrapper is not None:
            # init data loader
            data_loader = DataLoaderWrapper(config)
        else:
            data_loader = None
        # init train excecutor
        executor = Train_Executor(config, data_loader)

        # Start training
        executor.train()

    elif config.mode == 'test':
        if DataLoaderWrapper is not None:
            # init data loader
            data_loader = DataLoaderWrapper(config)
        else:
            data_loader = None
        # init test executor
        evaluator = Test_Evaluator(config, data_loader)
        # Run Evaluation
        evaluator.evaluate()
        
    elif config.mode == 'select_turn':
        if DataLoaderWrapper is not None:
            # init data loader
            data_loader = DataLoaderWrapper(config)
        else:
            data_loader = None
        
        # init train excecutor
        from train_executor.turn_selector import TurnSelector
        Turn_Selector = TurnSelector
        executor = Turn_Selector(config, data_loader)
        
        # train an initial model using budget 
        if config.budget > 0:
            print()
            executor.budget_train(config.budget_epoch)
        
        # select turn to label for each dialogue
        print()
        executor.active_learning()
        

def initialization(args, env=None):
    assert args.mode in ['train', 'test', 'run', 'select_turn']
    # ===== Process Config =======
    config = process_config(args, env=env)
    if config is None:
        return None
    # Create Dirs
    dirs = [
        config.log_path,
    ]
    if config.mode == 'train':
        dirs += [
            config.saved_model_path,
            config.imgs_path,
            config.tensorboard_path
        ]
    if config.mode == 'test':
        dirs += [
            config.imgs_path,
            config.results_path,
        ]

    if config.reset and config.mode == "train":
        # Reset all the folders
        print("You are deleting following dirs: ", dirs, "input y to continue")
        if input() == 'y':
            for dir in dirs:
                try:
                    delete_dir(dir)
                except Exception as e:
                    print(e)
            # Reset load epoch after reset
            config.train.load_epoch = 0
        else:
            print("reset cancelled.")

    create_dirs(dirs)
    logger.init_logger(config)

    # set cuda flag
    is_cuda = torch.cuda.is_available()
    print(f'Detecting CUDA availability: {is_cuda}')
    print(f'config.cuda availability: {config.cuda}')
    if is_cuda and not config.cuda:
        print("WARNING: You have a CUDA device, so you should probably enable CUDA")

    cuda = is_cuda & config.cuda

    if cuda:
        config.device = torch.device("cuda")
        torch.cuda.set_device(config.gpu_device)
        config.cuda_device = config.gpu_device
        print("Program will run on *****GPU-CUDA***** ")
        print_cuda_statistics()
    else:
        config.cuda_device = -1
        config.device = torch.device("cpu")
        print("Program will run on *****CPU*****\n")
    
    #### UPDATE: add random seed
    n_gpu = 0
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
    print(f'Number of GPU available: {n_gpu}')
        
    print(f'Ramdom seed: {config.random_seed}')
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if n_gpu > 0:
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print('Initialization done with the config:', str(config))
    return config

def parse_args_sys(args_list=None, env=None):
    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="Knowledge-aware graph-enhanced GPT-2 for DST")
    
    # parse different args based on ENV
    if env == 'sagemaker_training_job':
        arg_parser.add_argument(
            '--config',
            type=str,
            default='None',
            help='The Configuration file in json/jsonnet format'
        )
        
        arg_parser.add_argument('--selected_turn_path',  type=str, default='results')
        
        # SageMaker parameters
        arg_parser.add_argument('--model_dir',        type=str, default='/opt/ml/model')
        arg_parser.add_argument('--output_data_dir',  type=str, default='/opt/ml/output/data')
        arg_parser.add_argument('--checkpoint_path',  type=str, default='/opt/ml/checkpoints')
        arg_parser.add_argument('--sm_tensorboard_path',  type=str, default='/opt/ml/output/tensorboard')
        arg_parser.add_argument('--non_spot_checkpoint_path',  type=str, default='/opt/ml/output/non_spot_checkpoints')
        # param that indicates if using spot training, default is False
        arg_parser.add_argument('--use_spot_instances',  action='store_true', default=False, help='Switch to use Spot training')
    else:
        arg_parser.add_argument(
            'config',
            metavar='config_json_file',
            default='None',
            help='The Configuration file in json/jsonnet format'
        )
    
    
    arg_parser.add_argument('--DATA_FOLDER', type=str, default='',  help='The path where the data is saved.')
    arg_parser.add_argument('--EXPERIMENT_FOLDER', type=str, default='', help='The path where the experiments will be saved.')
    arg_parser.add_argument('--disable_cuda', action='store_true', default=False, help='Disable CUDA, run on CPU.')
    arg_parser.add_argument('--device', type=int, default=-1, help='Which CUDA device to use. Device ID.')

    arg_parser.add_argument('--mode', type=str, default='', help='train/test, see README.md for more details.')
    arg_parser.add_argument('--reset', action='store_true', default=False, help='This flag will try to delete already generated experiment data.')
    
    ### Update here
    arg_parser.add_argument('--only_last_turn', action='store_true', default=False, help='Switch to use sparse supervision, 14.3 percent of data')
    arg_parser.add_argument('--dummy_dataloader', action='store_true', default=False, help='Use only a small portion of data to run the program. Useful for debugging.')
    
    arg_parser.add_argument('--experiment_name', type=str, default='', help='The experiment name of the current run.')
    arg_parser.add_argument('--fp16', action='store_true', default=False, help='Not used.')

    arg_parser.add_argument('--load_best_model', action='store_true', default=False, help='Whether to load best model for testing/continue training.')
    arg_parser.add_argument('--load_epoch', type=int, default=-1, help='Specify which epoch of model to load from.')
    arg_parser.add_argument('--load_model_path', type=str, default="", help='Specify a path to a pre-trained model.')

    arg_parser.add_argument('--test_num_evaluation', type=int, default=-1, help='How many data entries need to be tested.')
    arg_parser.add_argument('--test_batch_size', type=int, default=-1, help='Batch size of test.')
    arg_parser.add_argument('--test_num_processes', type=int, default=-1, help='0 to disable multiprocessing testing; default is 4.')
    arg_parser.add_argument('--test_evaluation_name', type=str, default="", help='Evaluation name which will be created at /path/to/experiment/test/$test_evaluation_name$')
    arg_parser.add_argument('--test_disable_plot_img', action='store_true', default=False, help='Not used.')
    arg_parser.add_argument('--test_output_attention', action='store_true', default=False, help='For extracting attention scores. No effect for reproduction.')

    arg_parser.add_argument('--num_head', type=int, default=4, help='Number of attention heads of GATs')
    arg_parser.add_argument('--num_layer', type=int, default=4, help='Number of GAT layers')
    arg_parser.add_argument('--num_hop', type=int, default=2, help='Number of GAT hops.')
    arg_parser.add_argument('--graph_mode', type=str, default="part", help='part: DSGraph; full: DSVGraph')

    # add random seed
    arg_parser.add_argument('--random_seed', type=int, default=42, help='random seed')
    arg_parser.add_argument('--save_results_path',  type=str, default='results')
    arg_parser.add_argument('--budget',  type=int, default=0, help='The number of labelled turns initially')
    arg_parser.add_argument('--budget_random_turn',  type=int, default=0, help='Wether to randomly choice a turn from a dialogue')
    arg_parser.add_argument('--budget_epoch', type=int, default=5, help='Number of epochs to train the budget')
    arg_parser.add_argument('--round_epoch', type=int, default=5, help='Number of epochs to train the after each round')
    
    arg_parser.add_argument('--acquisition', type=str, default='max_entropy', help='Choose an acquisition function to be used for AL')
    
    # to continue select turn
    arg_parser.add_argument('--continue_select_turn',  type=int, default=0, help='continue select turn')
    
    if args_list is None:
        args = arg_parser.parse_args()
    else:
        args = arg_parser.parse_args(args_list)
    return args

if __name__ == '__main__':
    # get ENV
    env = os.getenv('ENV')
    
    args = parse_args_sys(env=env)
    print(args)
    config = initialization(args, env=env)
    pprint(config)
    if config is None:
        exit(0)
        
    if env == 'sagemaker_training_job':
        print(f'Running using Sagemaker Training Job!')
        config['ENV'] = 'sagemaker_remote'
    else:
        config['ENV'] = None
        
    main(config)