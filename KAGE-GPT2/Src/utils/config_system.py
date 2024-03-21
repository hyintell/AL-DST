import os
import shutil
import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler

import json
import _jsonnet
import datetime
import time
from easydict import EasyDict
from pprint import pprint
import time
from utils.dirs import create_dirs
from pathlib import Path

def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file: the path of the config file
    :return: config(namespace), config(dictionary)
    """

    # parse the configurations from the config json file provided

    try:
        config_dict = json.loads(_jsonnet.evaluate_file(json_file))
        # EasyDict allows to access dict values as attributes (works recursively).
        config = EasyDict(config_dict)
        return config, config_dict
    except ValueError:
        print("INVALID JSON file.. Please provide a good json file")
        exit(-1)

def process_config(args, env=None):
    script_dir = os.path.dirname(os.path.realpath('__file__'))
    path = Path(script_dir).parent
    config, _ = get_config_from_json(args.config)
    
    if env == 'sagemaker_training_job':
        config.model_dir = args.model_dir
        print(f'model_dir is: {config.model_dir}')
        
        ############### checkpoint_path: =>/opt/ml/checkpoints
        config.checkpoint_path = args.checkpoint_path
        print(f'checkpoint_path is: {config.checkpoint_path}')

        config.output_data_dir = args.output_data_dir
        print(f'output_data_dir is: {config.output_data_dir}')

        config.non_spot_checkpoint_path = args.non_spot_checkpoint_path
        print(f'non_spot_checkpoint_path is: {config.non_spot_checkpoint_path}')

        config.sm_tensorboard_path = args.sm_tensorboard_path
        print(f'sm_tensorboard_path is: {config.sm_tensorboard_path}')

        config.use_spot_instances = args.use_spot_instances
        print(f'use_spot_instances is: {config.use_spot_instances}')

        config.selected_turn_path = args.selected_turn_path

    # Some default paths
    if not config.DATA_FOLDER:
        # Default path
        config.DATA_FOLDER = os.path.join(str(path), 'Data')
    if not config.EXPERIMENT_FOLDER:
        # Default path
        config.EXPERIMENT_FOLDER = os.path.join(str(path), 'Experiments')
    if not config.TENSORBOARD_FOLDER:
        # Default path
        config.TENSORBOARD_FOLDER = os.path.join(str(path), 'Data_TB')

    # Override using passed parameters
    config.cuda = not args.disable_cuda
    if args.device != -1:
        config.gpu_device = args.device
    config.reset = args.reset
    config.mode = args.mode
    if args.experiment_name != '':
        config.experiment_name = args.experiment_name
    config.model_config.graph_model.num_layer = args.num_layer
    config.model_config.graph_model.num_head = args.num_head
    config.model_config.graph_model.num_hop = args.num_hop
    config.model_config.graph_mode = args.graph_mode
    config.random_seed = args.random_seed
    config.save_results_path = args.save_results_path
    config.budget = args.budget
    config.budget_random_turn = args.budget_random_turn
    config.budget_epoch = args.budget_epoch
    config.round_epoch = args.round_epoch
    config.acquisition = args.acquisition
    
    config.continue_select_turn = args.continue_select_turn
    print(f'save_results_path is: {config.save_results_path}')
    print(f'budget is: {config.budget}')
    print(f'budget_random_turn is: {config.budget_random_turn}')
    print(f'budget_epoch is: {config.budget_epoch}')
    print(f'round_epoch is: {config.round_epoch}')
    print(f'acquisition is: {config.acquisition}')
    
    print(f'continue_select_turn is: {config.continue_select_turn}')
    
    # Override using args for only last turn...
    config.data_loader.additional.only_last_turn = args.only_last_turn

    config.data_loader.dummy_dataloader = args.dummy_dataloader
    config.fp16 = args.fp16
    config.test.plot_img = not args.test_disable_plot_img
    if args.test_num_evaluation != -1:
        config.test.num_evaluation = args.test_num_evaluation
    if args.test_batch_size != -1:
        config.test.batch_size = args.test_batch_size
    if args.test_evaluation_name:
        config.test.evaluation_name = args.test_evaluation_name
    config.test_output_attention = args.test_output_attention
    if args.test_num_processes != -1:
        config.test.additional.multiprocessing = args.test_num_processes

    if config.mode == "train":
        print(f'args.load_best_model: {args.load_best_model}')
        config.train.load_best_model = args.load_best_model
        config.train.load_model_path = args.load_model_path
        config.train.load_epoch = args.load_epoch
    else:
        config.test.load_best_model = args.load_best_model
        config.test.load_model_path = args.load_model_path
        config.test.load_epoch = args.load_epoch

    # Generated Paths
    config.log_path = os.path.join(config.EXPERIMENT_FOLDER, config.experiment_name, config.mode)
    config.experiment_path = os.path.join(config.EXPERIMENT_FOLDER, config.experiment_name)
    config.saved_model_path = os.path.join(config.EXPERIMENT_FOLDER, config.experiment_name, "train", 'saved_model')
    
    if config.mode == "train":
        config.imgs_path = os.path.join(config.EXPERIMENT_FOLDER, config.experiment_name, "train", 'imgs')
    else:
        config.imgs_path = os.path.join(config.EXPERIMENT_FOLDER, config.experiment_name, "test",
                                        config.test.evaluation_name, 'imgs')
#         if env == 'sagemaker_training_job':
        config.results_path = os.path.join(config.save_results_path, config.experiment_name, f'RandomSeed-{config.random_seed}')
#         else:
#             config.results_path = os.path.join(config.EXPERIMENT_FOLDER, config.experiment_name, "test",
#                                             config.test.evaluation_name)
    config.tensorboard_path = os.path.join(config.TENSORBOARD_FOLDER, config.experiment_name)

    return config


