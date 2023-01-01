#!/usr/bin/env bash

# Experiment Group 1
# python3.8 train_two_stream.py -e 20 -m freeze_ag2b32_8video_dp0.2_step_aug_weighted_loss_lr1e-4 -f -ag 2 -cmode Step -tv 1 2 4 7 8 10 11 12 -vv 5 6 -cpe 5 -dp 0.2
# python3.8 train_two_stream.py -e 20 -m freeze_ag2b32_8video_dp0.2_step_aug_weighted_loss_lr1e-5 -f -ag 2 -cmode Step -tv 1 2 4 7 8 10 11 12 -vv 5 6 -cpe 5 -dp 0.2 -lr 0.00001
# 
# python3.8 train_two_stream.py -e 20 -m freeze_ag2b32_8video_dp0.5_step_aug_weighted_loss_lr1e-4 -f -ag 2 -cmode Step -tv 1 2 4 7 8 10 11 12 -vv 5 6 -cpe 5 -dp 0.5
# python3.8 train_two_stream.py -e 20 -m freeze_ag2b32_8video_dp0.5_step_aug_weighted_loss_lr1e-5 -f -ag 2 -cmode Step -tv 1 2 4 7 8 10 11 12 -vv 5 6 -cpe 5 -dp 0.5 -lr 0.00001
# 
# python3.8 train_two_stream.py -e 20 -m unfreeze_ag2b32_8video_dp0.2_step_aug_weighted_loss_lr1e-4 -ag 2 -cmode Step -tv 1 2 4 7 8 10 11 12 -vv 5 6 -cpe 5 -dp 0.2
# python3.8 train_two_stream.py -e 20 -m unfreeze_ag2b32_8video_dp0.2_step_aug_weighted_loss_lr1e-5 -ag 2 -cmode Step -tv 1 2 4 7 8 10 11 12 -vv 5 6 -cpe 5 -dp 0.2 -lr 0.00001
# 
# python3.8 train_two_stream.py -e 20 -m unfreeze_ag2b32_8video_dp0.5_step_aug_weighted_loss_lr1e-4 -ag 2 -cmode Step -tv 1 2 4 7 8 10 11 12 -vv 5 6 -cpe 5 -dp 0.5
# python3.8 train_two_stream.py -e 20 -m unfreeze_ag2b32_8video_dp0.5_step_aug_weighted_loss_lr1e-5 -ag 2 -cmode Step -tv 1 2 4 7 8 10 11 12 -vv 5 6 -cpe 5 -dp 0.5 -lr 0.00001
# 
# lr=1e-4 is too large, 1e-5 works better
# dropout = 0.5 works better than dropout = 0.2
# unfreeze works better than freeze
# 
# Experiment Group 2
# python3.8 train_two_stream.py -e 20 -m unfreeze_ag2b32_8video_dp0.7_step_aug_weighted_loss_lr1e-5 -ag 2 -cmode Step -tv 1 2 4 7 8 10 11 12 -vv 5 6 -cpe 5 -dp 0.7 -lr 0.00001
# python3.8 train_two_stream.py -e 20 -m unfreeze_ag4b32_8video_dp0.5_step_aug_weighted_loss_lr1e-5 -ag 4 -cmode Step -tv 1 2 4 7 8 10 11 12 -vv 5 6 -cpe 5 -dp 0.5 -lr 0.00001
# 
# droupout = 0.7 has little improvement
# accumulate gradient for 4 steps does not improve performance
# 
# Experiment Group 3
# python3.8 train_two_stream.py -e 20 -m unfreeze_ag2b32_8video_dp0.5_task_aug_weighted_loss_lr1e-5 -ag 2 -cmode Task -tv 1 2 4 7 8 10 11 12 -vv 5 6 -cpe 5 -dp 0.5 -lr 0.00001
#
# training for tasks does not have good results
#
# Experiment Group 4
# MLE python3.8 train_two_stream.py -e 20 -m unfreeze_ag2b32_8video_dp0.5_step_aug_weighted_loss_lr1e-5_vgg16_bn -ag 2 -cmode Step -tv 1 2 4 7 8 10 11 12 -vv 5 6 -cpe 5 -dp 0.5 -lr 0.00001 -bb vgg16_bn
# TODO: python3.8 train_two_stream.py -e 20 -m unfreeze_ag2b32_8video_dp0.5_step_aug_weighted_loss_lr1e-5_cj_0.7 -ag 2 -cmode Step -tv 1 2 4 7 8 10 11 12 -vv 5 6 -cpe 5 -dp 0.5 -lr 0.00001 -cj 0.7
# TODO: python3.8 train_two_stream.py -e 20 -m unfreeze_ag2b32_8video_dp0.5_step_aug_weighted_loss_lr1e-5_vgg19 -ag 2 -cmode Step -tv 1 2 4 7 8 10 11 12 -vv 5 6 -cpe 5 -dp 0.5 -lr 0.00001 -bb vgg19
# Experiment Group 5
# python3.8 train_two_stream.py -e 20 -m cv_4_5 -tv 1 2 6 7 8 10 11 12 14 15 -vv 4 5
# python3.8 train_two_stream.py -e 20 -m cv_14_15 -tv 1 2 4 5 6 7 8 10 11 12 -vv 14 15
# python3.8 train_two_stream.py -e 20 -m cv_11_12 -tv 1 2 4 5 6 7 8 10 14 15 -vv 11 12
# python3.8 train_two_stream.py -e 20 -m cv_8_10 -tv 1 2 4 5 6 7 11 12 14 15 -vv 8 10
# python3.8 train_two_stream.py -e 20 -m cv_6_7 -tv 1 2 4 5 8 10 11 12 14 15 -vv 6 7
# python3.8 train_two_stream.py -e 20 -m cv_1_2 -tv 4 5 6 7 8 10 11 12 14 15 -vv 1 2
#
# python3.8 train_two_stream.py -e 3 -m cv_t2_14_15 -tv 1 2 14 5 6 7 8 10 11 12 -vv 4 15
#
# python3.8 train_two_stream.py -e 3 -m cv_t2_14 -tv 1 2 4 5 6 7 8 10 11 12 15 -vv 14
# python3.8 train_two_stream.py -e 3 -m cv_10videos_1_5 -tv 2 4 6 7 8 10 11 12 -vv 1 5
# python3.8 train_two_stream.py -e 3 -m cv_10videos_2_4 -tv 1 5 6 7 8 10 11 12 -vv 2 4
# python3.8 train_two_stream.py -e 3 -m cv_10videos_6_7 -tv 1 2 4 5 8 10 11 12 -vv 6 7
# python3.8 train_two_stream.py -e 3 -m cv_10videos_8_10 -tv 1 2 4 5 6 7 11 12 -vv 8 10
# python3.8 train_two_stream.py -e 3 -m cv_10videos_11_12 -tv 1 2 4 5 6 7 8 10 -vv 11 12
#
# python3.8 train_two_stream.py -e 15 -ckpt ./runs/cv_10videos_1_5_20221229_223039/ckpts/ckpt_epoch_05.ckpt
# python3.8 train_two_stream.py -e 15 -ckpt ./runs/cv_10videos_2_4_20221229_225821/ckpts/ckpt_epoch_11.ckpt
# python3.8 train_two_stream.py -e 15 -ckpt ./runs/cv_10videos_6_7_20221229_232618/ckpts/ckpt_epoch_05.ckpt
# python3.8 train_two_stream.py -e 15 -ckpt ./runs/cv_10videos_8_10_20221229_235359/ckpts/ckpt_epoch_05.ckpt
# python3.8 train_two_stream.py -e 15 -ckpt ./runs/cv_10videos_11_12_20221230_002145/ckpts/ckpt_epoch_05.ckpt

# python3.8 train_two_stream.py -e 3 -m cv_10videos_2_8 -tv 1 4 5 6 7 10 11 12 -vv 2 8
# python3.8 train_two_stream.py -e 3 -m cv_10videos_4_8 -tv 1 2 5 6 7 10 11 12 -vv 4 8
# python3.8 train_two_stream.py -e 3 -m cv_10videos_2_4_1.0_1.05_1.0 -tv 1 5 6 7 8 10 11 12 -vv 2 4
#
python3.8 train_two_stream.py -e 12 -ckpt ./runs/cv_10videos_2_4_1.0_1.05_1.0_20230101_025323/ckpts/ckpt_epoch_05.ckpt



