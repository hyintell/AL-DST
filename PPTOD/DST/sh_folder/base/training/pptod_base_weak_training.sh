CUDA_VISIBLE_DEVICES=0 python ../../../learn.py\
    --model_name t5-base\
    --epoch_num 36\
    --gradient_accumulation_steps 4\
    --number_of_gpu 1\
    --batch_size_per_gpu 4\
    --data_path_prefix ../../../../data/multiwoz/data/multi-woz-fine-processed/\
    --pretrained_path ../../../../checkpoints/base/\
    --patience 3\
    --train_data_ratio 1.0\
    --ckpt_save_path ../../../ckpt/base/k500_S123\
    --only_last_turn 'True'\
    --only_random_turn 'False'\
    --random_seed 123
    
