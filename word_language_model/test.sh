#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
# python main.py --cuda --dropout 0.2 --epochs 40 --tied --sparsity 0.9 --prune --retrain  
python main.py --cuda --dropout 0.2 --epochs 40 --tied --sparsity 0.9 --prune --retrain --pretrained model.pt-pretrain-wikitext-2-LSTM2-200-200dropout0.20-tied
# python main.py --cuda --dropout 0.4 --epochs 6 --tied
# python main.py --cuda --dropout 0.4 --epochs 36 --tied
# python main.py --cuda --dropout 0.4 --epochs 30 --tied --sparsity 0.9 --retrain --pretrained model.pt-prune90.0-wikitext-2-LSTM2-200-200

