#!/usr/bin/env bash
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

ROOT=$PWD
FOLDER=ucf51_baseline
MODEL=r21d
DEPTH=18
VIDEO_PATH=datasets/UCF101/hdf5data
AUDIO_PATH=datasets/UCF101/ucf101_audiocnn14embed512_features

mkdir results/${FOLDER}

### training
python main.py \
--root_path ${ROOT} \
--video_path ${VIDEO_PATH} \
--audio_path ${AUDIO_PATH} \
--annotation_path datasets/UCF101/ucf51_01.json \
--result_path results/${FOLDER} \
--dataset ucf101 --n_classes 51 \
--model ${MODEL} --model_depth ${DEPTH} --batch_size 16 --n_threads 16 --checkpoint 100 \
--file_type hdf5 --sample_t_stride 1 --n_val_samples=3 \
--train_crop random --val_freq=1 \
--lr_scheduler multistep \
--learning_rate 0.001 --weight_decay 5e-4 \
--n_epochs=300

### testing recognition
python main.py \
--root_path ${ROOT} \
--video_path ${VIDEO_PATH} \
--annotation_path datasets/UCF101/ucf51_01.json \
--result_path results/${FOLDER} \
--resume_path results/${FOLDER}/save_model.pth \
--dataset ucf101 --n_classes 51 \
--model ${MODEL} --model_depth ${DEPTH}  \
--file_type hdf5 --sample_t_stride 1 \
--n_threads 16 --no_train --no_val --inference --output_topk 5 --inference_batch_size 1

### print acc
python -m util_scripts.eval_accuracy datasets/UCF101/ucf51_01.json results/${FOLDER}/val.json --subset validation -k 1 --ignore
