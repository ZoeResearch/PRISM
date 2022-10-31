#!/bin/bash

env=base
data=spot_bin
#embed_model=("fasttext") #dell bgru --dell
#embed_model=("glove") #dell    bgru--on xp dell blstm--dell,passwd          gru--dell--ok
#embed_model=("elmo") #base      bgru--base,passwd  blstm--xp base,dell  gru--base,dell
embed_model=("w2v") #passwd    bgru-- passwd,dell    blstm--passwd        gru--passwd,base
#embed_model=("w2v" "glove") #base
#embed_model=("bert_seq") #mingwen2
network=("gru" "lstm" "blstm" "bgru")
#network=("gru" "lstm" "bgru")
#network=("blstm" "textcnn")
#gpu=0
proc_num=0
single_proc_per_gpu=2
running_gpu=2
#gpu_num=4
#number=0
#program_num_per_gpu=1
gpu=0

for i in ${embed_model[@]}
do
  for j in ${network[@]}
  do
    echo "Start runnning ${i}${j} on gpu${gpu}..."
    echo "CUDA_VISIBLE_DEVICES=${gpu} nohup python -u sort_bug.py -f 0 -b 1 -t 1 -m embedding -c src -e ${i} -d ${j} > errlog/${i}_${j}.log 2>&1 &"
    CUDA_VISIBLE_DEVICES=${gpu} nohup python -u sort_bug.py -f 0 -b 1 -t 1 -m embedding -c src -e ${i} -d ${j} > errlog/${i}_${j}.log 2>&1 &
    procnum=`ps -ef|grep "python -u sort_bug.py -f 0 -b 1 -t 1 -m embedding -c src -e ${i} -d ${j}" |grep -v grep|wc -l`
    if [ $procnum -ne 0 ]; then
      echo "start successfully!"
    fi

    let proc_num++
    temp=$((${running_gpu}*${single_proc_per_gpu}))
    if [ ${proc_num} -ge ${temp} ]; then
      let running_gpu++
      gpu=$((${running_gpu}-1))
    fi
  done
done
