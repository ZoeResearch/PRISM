#!/bin/bash

env=base
#embed_model=elmo
data=spot_bin
embed_model=("w2v" "glove" "fasttext" "elmo" "bert")
#network=("textcnn" "mlp" "gru" "blstm")
network=("gru")
gpu=0
gpu_num=4
#number=0
#program_num_per_gpu=1

conda activate ${env}

for i in ${embed_model[@]}
do
  for j in ${embed_model[@]}
  do
    for k in ${embed_model[@]}
    do
      while true
      do
        echo "Start runnning src ${i} ir ${j} byte ${k}..."
        echo -e "CUDA_VISIBLE_DEVICES=$gpu python -u ./entry.py -e ${i} -i ${j} -b ${k} -r False -s True -d dir -c add > errlog/hybrid_add_${i}_${j}_${k}_${network}.log 2>&1"
        tmux new-window -n add_${i}_${j}_${k} \
        "CUDA_VISIBLE_DEVICES=$gpu python -u ./entry.py -e ${i} -i ${j} -b ${k} -r False -s True -d dir -c add > errlog/hybrid_add_${i}_${j}_${k}_${network}.log 2>&1"
        let gpu++
        sleep 20m
        procnum=`ps -ef|grep "python -u ./entry.py -e ${i} -i ${j} -b ${k} -r False -s True -d dir -c add" |grep -v grep|wc -l`
        if [ $procnum -ne 0 ]; then
          break     #说明启动成功
        fi
    # 没有识别到这个等式  导致无法启动gpu
        if ((${gpu}==${gpu_num})); then
          gpu=0
          echo "Waiting for available GPU..."
          sleep 1h
    #      let gpu=0
        fi
      done
    done
  done
done




