#!/bin/bash

cls=("0")
emb=("w2v" "fasttext" "glove" "elmo")

for i in ${cls[@]}
do
    for j in ${emb[@]}
    do
        echo "python sample_TSNE.py -c ${i} -e ${j}"
        nohup python -u sample_TSNE.py -c ${i} -e ${j} > ${i}_${j}.log &
    done
done
