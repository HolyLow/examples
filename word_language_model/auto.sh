#!/bin/bash
prunelog="rnn_prun_warmup.log"
originlog="rnn.log"
cmd="python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40 --tied"

$cmd --prune >$prunelog 2>&1
# $cmd >$originlog 2>&1
