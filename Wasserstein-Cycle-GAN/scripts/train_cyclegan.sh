#!/bin/bash
out=$'/console.out'
ckpt_dir=$'./checkpoints/'
date=$(date +%y-%m-%d-%H-%M)    
modelname=$'_Wcyclegan'

mkdir "./checkpoints/$date$modelname/"

nohup \
python train.py 	\
						--dataroot ./datasets/comic \
						--name "$date$modelname" \
						--model cycle_gan \
						--batchSize 20 \
						--input_nc 1 \
						--output_nc 1 \
						--niter 250 \
						--niter_decay 25 \
						--epoch_count 1 \
						--which_model_netG resnet_9blocks \
						--ngf 32 \
						--which_model_netD n_layers \
						--ndf 16 \
						--n_layers_D 3 \
						--lr 1e-4 \
						--lambda_A 10 \
						--lambda_B 10 \
						--image_width 256 \
						--image_height 128 \
						--norm instance \
						--no_flip \
						--resize_or_crop scale \
						--norm batch \
> "./checkpoints/$date$modelname/console.out" & 

