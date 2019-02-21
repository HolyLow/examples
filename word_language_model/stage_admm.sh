#!/bin/bash
config_yaml="tmp.yaml"
ExportConfig() {
  config_sp=`echo "scale=4;$1/100" | bc`
  echo -e "ptb:\n  prune_ratios:\n    rnn.weight_ih_l0:\n      ${config_sp}\n    rnn.weight_hh_l0:\n      ${config_sp}\n    rnn.weight_ih_l1:\n      ${config_sp}\n    rnn.weight_hh_l1:\n      ${config_sp}" > $2

}

begin_sp=50
end_sp=90
sp=$begin_sp

log="sp${begin_sp}-${end_sp}_stage_admm_tied_1500.log"
rm $log &>/dev/null

base="python -u admm_prune.py --cuda --emsize 1500 --nhid 1500 --data ./data/ptb --arch ptb --config_file $config_yaml --admm --rho 0.001 --sparsity_type balanced_row"
# load="ptb_pretrained_1500.pt"
load=" "
while [ $sp -le $end_sp ]; do
  if [ $sp -lt 70 ]; then
    epoch=10
    dropout=0.60
    lr=20
  elif [ $sp -lt 80 ]; then
    epoch=20
    dropout=0.55
    lr=10
  else
    epoch=40
    dropout=0.50
    lr=10
  fi
  save="sp${sp}_epoch${epoch}_dropout${dropout}_lstm_tied1500.pt"
  ExportConfig $sp ${config_yaml} 
  cmd="$base --epoch $epoch --dropout $dropout --save $save $load"
  echo "$cmd" | tee -a $log
  $cmd | tee -a $log
  if [ $sp -lt 70 ]; then
    change=5
  elif [ $sp -le 80 ]; then
    change=2
  else
    change=2
  fi
  sp=`expr $sp + $change`
  load="--pretrained $save"
done
