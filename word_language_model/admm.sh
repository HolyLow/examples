#!/bin/bash

CUDA_VISIBLE_GPUS=1 python -u admm_prune.py --cuda --emsize 1500 --nhid 1500 --dropout 0.40 --epochs 40 --tied --data ./data/ptb --arch ptb --config_file lstm.yaml --admm --rho 0.001 --sparsity_type balanced_row | tee admm_1500_tied_rho1E-3_drop40.log
# python -u admm_prune.py --cuda --emsize 1500 --nhid 1500 --dropout 0.60 --epochs 40 --tied --data ./data/ptb --arch ptb --config_file lstm.yaml --admm --rho 0.001 --sparsity_type balanced_row | tee admm_1500_tied_rho1E-3_drop60.log

# CUDA_VISIBLE_GPUS=3 python -u admm_prune.py --cuda --epochs 20 --data ./data/ptb --arch ptb --config_file lstm.yaml --admm --rho 0.01 --sparsity_type balanced_row
