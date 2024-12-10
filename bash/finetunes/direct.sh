export CUDA_VISIBLE_DEVICES=0

python freeze_finetune.py \
--model_name gpt2 \
--data_dir Templama \
--results_dir temporal_res/1118_3 \
--method direct