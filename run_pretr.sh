#!/bin/bash
export DATA_DIR=data
#python3 build_openwebtext_pretraining_dataset.py --max-seq-length 512 --data-dir $DATA_DIR --num-processes 5
python3 run_pretraining.py --data-dir $DATA_DIR --model-name electra_small_owt_v10.l512.verify --hparams '{"model_size": "small", "num_train_steps": 100000, "mask_prob": 0.1, "rich_prob": 0.5, "use_bilm": "False", "num_preds": 4, "max_seq_length": 512, "train_batch_size":32}'
wait
python3 run_finetuning.py --data-dir data --model-name electra_small_owt_v10.l512.verify --hparams '{"model_size": "small", "task_names": ["squad"]}'
wait
python3 run_finetuning.py --data-dir data --model-name electra_small_owt_v10.l512.verify --hparams '{"model_size": "small", "task_names": ["squadv1"]}'
wait
python3 run_finetuning.py --data-dir data --model-name electra_small_owt_v10.l512.verify --hparams '{"model_size": "small", "task_names": ["chunk"]}'
wait
python3 run_finetuning.py --data-dir data --model-name electra_small_owt_v10.l512.verify --hparams '{"model_size": "small", "task_names": ["triviaqa"]}'
