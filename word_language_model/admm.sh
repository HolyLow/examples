#!/bin/bash

python -u admm_prune.py --cuda --emsize 1500 --nhid 1500 --dropout 0.50 --epochs 40 --tied --data ./data/ptb --arch ptb --config_file lstm.yaml --admm --rho 0.01 --sparsity_type balanced_row | tee admm_1500_tied.log

# CUDA_VISIBLE_GPUS=3 python -u admm_prune.py --cuda --epochs 20 --data ./data/ptb --arch ptb --config_file lstm.yaml --admm --rho 0.01 --sparsity_type balanced_row
