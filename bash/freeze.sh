export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9

python freeze_finetune.py \
--run_mode finetune \
--results_dir temporal_res/llama7b_1226 \
--model_name /home/chenyuheng/KN2/Llama/Llama7bChat \
--saved_models_dir saved_models/LLaMA/epoch50 \
--epochs 50 \
> temporal_res/llama7b_1226/xshell_output/freeze_finetune.txt 2>&1
