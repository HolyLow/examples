#!/bin/bash

log="wt2_baseline.log"
export CUDA_VISIBLE_DEVICES=2
cmd="python main.py --cuda --epochs 6"
extra="--save model/baseline/model.pt"
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

