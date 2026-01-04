#!/bin/bash

PYTHON=python3.10


# ==========================================
# 训练参数配置
# ==========================================
# Base Model (小)
BASE_HEADS=5
BASE_OPTIMAL_LR=1e-3
BASE_OPTIMAL_NOISE=1.0531207066658843

# Target Model (大)
TARGET_LAYER=10
TARGET_HEADS=20 # Width = 1280
TOTAL_STEPS=1000

# Batch & Accum
BATCH_SIZE=16
GRAD_ACCUM=8

echo "🚀 Starting muP Training..."

$PYTHON -m train \
    --n_layer $TARGET_LAYER \
    --n_head $TARGET_HEADS \
    --n_head_base $BASE_HEADS \
    --base_optimal_lr $BASE_OPTIMAL_LR \
    --base_optimal_noise $BASE_OPTIMAL_NOISE \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --total_steps $TOTAL_STEPS \
    --out_dir "out_muP_H${TARGET_HEADS}" \
    --per_sample_clip \
    --wandb \
    --init_from "scratch"
