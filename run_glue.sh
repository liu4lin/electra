source ~/venv3.5.2-tf.1.15.0/bin/activate
python3 run_finetuning.py --data-dir data --model-name electra_small_owt_v10 --hparams '{"model_size": "small", "task_names": ["cola"]}'
wait
#python3 run_finetuning.py --data-dir data --model-name electra_small_owt_v10 --hparams '{"model_size": "small", "task_names": ["sst"]}'
python3 run_finetuning.py --data-dir data --model-name electra_small_owt_v10 --hparams '{"model_size": "small", "task_names": ["mrpc"]}'
wait
python3 run_finetuning.py --data-dir data --model-name electra_small_owt_v10 --hparams '{"model_size": "small", "task_names": ["sts"]}'
wait
python3 run_finetuning.py --data-dir data --model-name electra_small_owt_v10 --hparams '{"model_size": "small", "task_names": ["qqp"]}'
wait
python3 run_finetuning.py --data-dir data --model-name electra_small_owt_v10 --hparams '{"model_size": "small", "task_names": ["mnli"]}'
wait
python3 run_finetuning.py --data-dir data --model-name electra_small_owt_v10 --hparams '{"model_size": "small", "task_names": ["qnli"]}'
wait
python3 run_finetuning.py --data-dir data --model-name electra_small_owt_v10 --hparams '{"model_size": "small", "task_names": ["rte"]}'
