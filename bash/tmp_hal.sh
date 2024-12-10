#export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9

#python freeze_finetune.py \
#--run_mode overlap \
#--results_dir /home/chenyuheng/KN2/kn2/temporal_res/llama7b_1226/res_wo_acc \
#--model_name /home/chenyuheng/KN2/Llama/Llama7bChat
python freeze_finetune.py \
--run_mode overlap \
--results_dir temporal_res/1118_3 \
--model_name gpt2 \
--saved_models_dir saved_models/epoch100