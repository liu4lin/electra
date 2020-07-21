source ~/venv3.5.2-tf.1.15.0/bin/activate
python3 run_finetuning.py --data-dir data --model-name electra_small_owt_v10 --hparams '{"model_size": "small", "task_names": ["squad"]}'
