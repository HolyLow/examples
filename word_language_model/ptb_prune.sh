#!/bin/bash

log="ptb_prune.log"
export CUDA_VISIBLE_DEVICES=3
cmd="python main.py --cuda --epochs 6"
extra="--save model/prune/model.pt --data ./data/ptb --prune --retrain --sparsity 0.9"
rm $log
echo $cmd $extra >>$log
$cmd $extra >>$log

cmd="python main.py --cuda --epochs 6 --tied"
echo $cmd $extra >>$log
$cmd $extra >>$log

cmd="python main.py --cuda --tied"
echo $cmd $extra >>$log
$cmd $extra >>$log

cmd="python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40"
echo $cmd $extra >>$log
$cmd $extra >>$log

cmd="python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied"
echo $cmd $extra >>$log
$cmd $extra >>$log

cmd="python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40"
echo $cmd $extra >>$log
$cmd $extra >>$log

cmd="python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40 --tied"
echo $cmd $extra >>$log
$cmd $extra >>$log

