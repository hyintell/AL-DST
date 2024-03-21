# STRATEGY='random'
# STRATEGY='least_confidence'
STRATEGY='max_entropy'

CUDA_VISIBLE_DEVICES=0 python ../../../active_learn.py\
    --model_name t5-base\
    --epoch_num 36\
    --gradient_accumulation_steps 4\
    --number_of_gpu 1\
    --batch_size_per_gpu 4\
    --data_path_prefix ../../../../data/multiwoz/data/multi-woz-fine-processed/\
    --pretrained_path ../../../../checkpoints/base/\
    --train_data_ratio 1.0\
    --patience 4\
    --ckpt_save_path ../../../ckpt/base/active_learn/LC-S413\
    --only_last_turn 'False'\
    --only_random_turn 'False'\
    --acquisition $STRATEGY\
    --num_dialogue_per_round 2000\
    --random_seed 111
    
