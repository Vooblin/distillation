# Distillation

### Standard approach

To work with usual method from ["Distilling the Knowledge in a Neural Network"](https://arxiv.org/abs/1503.02531) you can use [specific function](./old_demo/distillation.py#L33). It works with keras models and you can run it with your own models and datasets. Also, there are some supported datasets and architectures, you can find it in [other file](./old_demo/utils.py).

This function also has some hyperparameters (temp, a, b), you can find good description [here](https://nervanasystems.github.io/distiller/knowledge_distillation.html).

### TinyBert approach

This method ([paper](https://arxiv.org/abs/1909.10351)) are only for [Transformer](https://arxiv.org/abs/1706.03762) architecture. Here it works with different Bert model versions ([Bert](https://arxiv.org/abs/1810.04805), [Roberta](https://arxiv.org/abs/1907.11692), [Albert](https://arxiv.org/abs/1909.11942), [TinyBert](https://arxiv.org/abs/1909.10351), [MobileBert](https://openreview.net/forum?id=SJxjVaNKwB)).

Supported tasks you can find in [dataset processors file](./prepare_dataset/dataset_processors.py).  
Examples of model configs you can find in [separate directory](./configs/).

How to use:
1. Prepare data for pretraining distillation  
   ```
   python3 prepare_dataset/create_distilled_pretraining_data.py \
     --input_file /path/to/input_file \
     --output_file /path/to/output_file \
     --vocab_file /path/to/vocabulary_file \
     --max_seq_length MSN (default 128)
   ```
2. Run pretraining distillation  
   ```
   python3 run_pretraining.py \
     --data_dir /path/to/pretraining_distillation_data \
     --output_dir /path/to/output_directory \
     --config_path /path/to/student_model_config \
     --teacher_config_path /path/to/teacher_model_config \
     --teacher_ckpt_path /path/to/teacher_model_checkpoint \
     --max_seq_length MSL \ (default 128)
     --batch_size BS \ (default 8)
     --epochs E \ (default 100)
     --steps_per_epoch SPE \ (default 10000)
     --steps_per_loop SPL \ (default 100)
     --warmup_steps WS \ (default 10000)
     --learning_rate LR (default 5e-5)
   ```
3. Prepare data for finetuning distillation
   ```
   python3 prepare_dataset/create_finetuning_data.py \
     --task_name task \
     --data_dir /path/to/data \
     --vocab_path /path/to/vocabulary_file \
     --output_dir /path/to/output_directory \
     --max_seq_length MSN (default 128)
   ```
4. Run finetuning distillation
   ```
   python3 run_finetuning.py \
     --task_name task \
     --data_dir \path\to\finetuning_distillation_data \
     --output_dir \path\to\output_directory \
     --config_path \path\to\student_model_config \
     --teacher_config_path \path\to\teacher_model_config \
     --ckpt_path \path\to\student_model_checkpoint \
     --teacher_ckpt_path \path\to\teacher_model_checkpoint \
     --max_seq_length MSL \ (default 128)
     --batch_size BS \ (default 8)
     --epochs E \ (default 3)
     --learning_rate LR (default 5e-5)
   ```
