#!/bin/bash

env=base
#embed_model=elmo
cls=("spot_bin" "spot_mul")
embed_model=("gru" "dpcnn")
#network=("textcnn" "mlp" "gru" "blstm")
hybrid=("3loss_src" "3loss_ir" "3loss_byte1" "3loss_byte2" "3loss_src_ir")
#network=("3loss_src_byte1" "3loss_src_byte2" "3loss_src_ir_byte1" "3loss_src_ir_byte2")
gpu=0
gpu_num=4
#number=0
#program_num_per_gpu=1

conda activate ${env}

for i in ${hybrid[@]}
do
  for j in ${cls[@]}
  do
    for k in ${embed_model[@]}
    do
      if (${i}=="3loss_src"]); then

      fi
      while true
      do
        echo "Start runnning hybrid ${i} cls ${j} embed ${k}..."
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




