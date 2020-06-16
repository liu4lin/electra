#!/bin/bash
export DATA_DIR=data
#python3 build_openwebtext_pretraining_dataset.py --data-dir $DATA_DIR --num-processes 5
nohup python3 run_pretraining.py --data-dir $DATA_DIR --model-name electra_small_owt --hparams '{"model_size": "small", "num_train_steps": 100000}' &
