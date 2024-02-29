#!/bin/bash
seed=401
run_dir="run_final"
model_type="RCAN" # options = {"RDN", "RCAN", "ESRT"}
topo_inclusion="none"
mode="evaluate" # options = {"train", "evaluate", "optuna_search"}

if [ ! -d "runs/${run_dir}" ]; then
    mkdir "runs/${run_dir}"
fi

if [ $mode="train" ]
then
    python3 main.py --gpu=0 --run_dir="runs/${run_dir}" --seed=$seed --topo_inclusion=$topo_inclusion --model_type=$model_type train
fi
if [ $mode="optuna_search" ]
then
    python3 main.py --gpu=0 --run_dir="runs/${run_dir}" --seed=$seed --topo_inclusion=$topo_inclusion --model_type=$model_type optuna_search
fi
if [ $mode="evaluate" ]
then
    python3 main.py --gpu=0 --run_dir="runs/${run_dir}" --seed=$seed --topo_inclusion=$topo_inclusion --model_type=$model_type evaluate
fi