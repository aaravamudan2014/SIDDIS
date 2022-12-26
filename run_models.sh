#!/bin/bash
seed=401
run_dir="run_final"
model_type="RCAN" # options = {"RDN", "RCAN"}
topo_inclusion="none" # options = {"none", "horizontal", "vertical", "beggining", "combination"}
mode="evaluate" # options = {"train", "evaluate", "optuna_search", "comparison"}

if [ ! -d "runs/${run_dir}" ]; then
    mkdir "runs/${run_dir}"
fi

# python3 main.py --gpu=0 --run_dir="runs/${run_dir}" --seed=$seed --topo_inclusion=$topo_inclusion --model_type=$model_type train
# python3 main.py --gpu=0 --run_dir="runs/${run_dir}" --seed=$seed --topo_inclusion=$topo_inclusion --model_type=$model_type optuna_search
python3 main.py --gpu=0 --run_dir="runs/${run_dir}" --seed=$seed --topo_inclusion=$topo_inclusion --model_type=$model_type evaluate
# python3 main.py --gpu=0 --run_dir="runs/${run_dir}" --seed=$seed --topo_inclusion=$topo_inclusion --model_type=$model_type comparison


# python3 main_combination.py --gpu=0 --run_dir="runs/${run_dir}" --seed=$seed --topo_inclusion=$topo_inclusion --model_type=$model_type train
# python3 main_combination.py --gpu=0 --run_dir="runs/${run_dir}" --seed=$seed --topo_inclusion=$topo_inclusion --model_type=$model_type evaluate
