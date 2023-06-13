#!/bin/bash

task(){
    sleep 0.5;
    echo "CUDA_VISIBLE_DEVICES=$1  python3 s_model_faster.py"
    CUDA_VISIBLE_DEVICES=$1  python3 s_model_faster.py --load_embeddings --no_last_evaluation --no_tqdm --random_seed --epochs 40
}


CUDA_VISIBLE_DEVICES=0  python3 s_model_faster.py --random_seed --epochs 40

M=4*100
N=4
(
for ((i=0; i<M; i++))
do
    ((i%N==0)) && wait
    echo $i
    j=$((i%N))
    task "$j" &
done
)

