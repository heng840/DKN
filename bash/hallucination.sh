export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9
#TODO 最后一组幻觉数据：/home/chenyuheng/KN2/kn2/Hallucination_res/llama/USE_complex_text
python hallucination.py \
--model_name /home/chenyuheng/KN2/Llama/Llama7bChat \
--use_complex_text \
--data_file datasets/wrong_fact_dataset/temporal/train-llama-complex/lama.jsonl \
--wrong_fact_dir datasets/wrong_fact_dataset/temporal/train-llama-complex \
--neurons_result_dir temporal_res/llama7b_1226/res_wo_acc \
--hallucination_parent_dir Hallucination_res/llama/USE_complex_text_0202 \
--run_mode train_test_split_check \
> /home/chenyuheng/KN2/kn2/temporal_res/llama7b_1226/xshell_output/hallucination_USE_complex_text.json 2>&1

python freeze_finetune.py \
--run_mode eval_only_direct \
--results_dir /home/chenyuheng/KN2/kn2/temporal_res/llama7b_1226/res_wo_acc \
--model_name /home/chenyuheng/KN2/Llama/Llama7bChat \
--saved_models_dir saved_models/LLaMA/epoch50 \
--epochs 50 \
> /home/chenyuheng/KN2/kn2/run_output/Eval_direct.txt 2>&1
python freeze_finetune.py \
--run_mode finetune_dkn \
--results_dir /home/chenyuheng/KN2/kn2/temporal_res/llama7b_1226/res_wo_acc \
--model_name /home/chenyuheng/KN2/Llama/Llama7bChat \
--saved_models_dir saved_models/LLaMA/epoch50 \
--epochs 50 \
> /home/chenyuheng/KN2/kn2/run_output/freeze_finetuneDKN.txt 2>&1
python freeze_finetune.py \
--run_mode finetune_kn \
--results_dir /home/chenyuheng/KN2/kn2/temporal_res/llama7b_1226/res_wo_acc \
--model_name /home/chenyuheng/KN2/Llama/Llama7bChat \
--saved_models_dir saved_models/LLaMA/epoch50 \
--epochs 50 \
> /home/chenyuheng/KN2/kn2/run_output/freeze_finetuneKN.txt 2>&1
python freeze_finetune.py \
--run_mode finetune_rnd \
--results_dir /home/chenyuheng/KN2/kn2/temporal_res/llama7b_1226/res_wo_acc \
--model_name /home/chenyuheng/KN2/Llama/Llama7bChat \
--saved_models_dir saved_models/LLaMA/epoch50 \
--epochs 50 \
> /home/chenyuheng/KN2/kn2/run_output/freeze_finetuneRND.txt 2>&1
python freeze_finetune.py \
--run_mode eval \
--results_dir /home/chenyuheng/KN2/kn2/temporal_res/llama7b_1226/res_wo_acc \
--model_name /home/chenyuheng/KN2/Llama/Llama7bChat \
--saved_models_dir saved_models/LLaMA/epoch50 \
--epochs 50 \
> /home/chenyuheng/KN2/kn2/run_output/Eval.txt 2>&1

#python hallucination.py \
#--model_name /home/chenyuheng/KN2/Llama/Llama7bChat \
#--data_file datasets/wrong_fact_dataset/temporal/train-llama/lama.jsonl \
#--wrong_fact_dir datasets/wrong_fact_dataset/temporal/train-llama \
#--neurons_result_dir temporal_res/llama7b_1226/res_wo_acc \
#--run_mode train_test_split_check \
#--hallucination_parent_dir Hallucination_res/llama/NOT_complex_text_130 \
#> /home/chenyuheng/KN2/kn2/temporal_res/llama7b_1226/xshell_output/hallucination_NOT_use_complex_text.json 2>&1
#
#
#python hallucination.py \
#--model_name /home/chenyuheng/KN2/Llama/Llama7bChat \
#--data_file datasets/wrong_fact_dataset/temporal/train-llama/lama.jsonl \
#--wrong_fact_dir datasets/wrong_fact_dataset/temporal/train-llama \
#--neurons_result_dir temporal_res/llama7b_1226/res_wo_acc \
#--run_mode not_split_check \
#--hallucination_parent_dir Hallucination_res/llama/NOT_complex_text_GOLDEN_130 \
#> /home/chenyuheng/KN2/kn2/temporal_res/llama7b_1226/xshell_output/hallucination_NOT_complex_text_GOLDEN.json 2>&1
#
#python hallucination.py \
#--model_name /home/chenyuheng/KN2/Llama/Llama7bChat \
#--use_complex_text \
#--data_file datasets/wrong_fact_dataset/temporal/train-llama-complex/lama.jsonl \
#--wrong_fact_dir datasets/wrong_fact_dataset/temporal/train-llama-complex \
#--neurons_result_dir temporal_res/llama7b_1226/res_wo_acc \
#--hallucination_parent_dir Hallucination_res/llama/USE_complex_text_GOLDEN \
#--run_mode not_split_check \
#> /home/chenyuheng/KN2/kn2/temporal_res/llama7b_1226/xshell_output/hallucination_USE_complex_text_GOLDEN.json 2>&1
#
#python hallucination.py \
#--model_name /home/chenyuheng/KN2/Llama/Llama7bChat \
#--use_complex_text \
#--data_file datasets/wrong_fact_dataset/temporal/train-llama-complex/lama.jsonl \
#--wrong_fact_dir datasets/wrong_fact_dataset/temporal/train-llama-complex \
#--neurons_result_dir temporal_res/llama7b_1226/res_wo_acc \
#--hallucination_parent_dir Hallucination_res/llama/USE_complex_text_GOLDEN \
#--run_mode not_split_check \
#> /home/chenyuheng/KN2/kn2/temporal_res/llama7b_1226/xshell_output/hallucination_USE_complex_text_GOLDEN.json 2>&1

#python hallucination.py \
#--use_complex_text \
#--model_name /home/chenyuheng/KN2/Llama/Llama7bChat \
#--data_file datasets/wrong_fact_dataset/temporal/train-llama-complex/lama.jsonl \
#--wrong_fact_dir datasets/wrong_fact_dataset/temporal/train-llama-complex \
#--neurons_result_dir temporal_res/llama7b_1226/res_wo_acc \
#> /home/chenyuheng/KN2/kn2/temporal_res/llama7b_1226/xshell_output/hallucination_use_complex_text.json 2>&1
#if [ ! -d "/home/chenyuheng/KN2/kn2/Hallucination_res/Xshell-output" ]; then
#    mkdir "/home/chenyuheng/KN2/kn2/Hallucination_res/Xshell-output"
#fi
#python hallucination.py \

#gpu_nums=2
#python -m torch.distributed.launch --nproc_per_node=$gpu_nums hallucination.py \
#--wrong_fact_dir datasets/wrong_fact_dataset/temporal/train-gpt2 \
#--neurons_result_dir temporal_res/1118_3 \
#--model_name gpt2 \
#--data_file datasets/wrong_fact_dataset/temporal/train-gpt2/lama.jsonl \
#> /home/chenyuheng/KN2/kn2/Hallucination_res/Xshell-output/GPT2_NOT_use_complex_text.json 2>&1
#gpu_nums=5
#python -m torch.distributed.launch --nproc_per_node=$gpu_nums hallucination.py \
#--wrong_fact_dir datasets/wrong_fact_dataset/temporal/train \
#--neurons_result_dir temporal_res/1118_3 \
#--model_name gpt2 \
#> run_output/hallucination.json 2>&1
#python -m torch.distributed.launch --nproc_per_node=$gpu_nums hallucination.py \
#--wrong_fact_dir datasets/wrong_fact_dataset/temporal/train \
#--neurons_result_dir temporal_res/1118_3 \
#--model_name gpt2 \
#--threshold 0.001 \
#--threshold_filter_DN 0.5 \
#> temporal_res/1118_3/run_output/hallucination.json 2>&1
#python -m torch.distributed.launch --nproc_per_node=$gpu_nums hallucination.py \
#--wrong_fact_dir datasets/wrong_fact_dataset/temporal/train \
#--neurons_result_dir temporal_res/1118_3 \
#--model_name gpt2 \
#--threshold 0.001 \
#--threshold_filter_DN 0.3 \
#> temporal_res/1118_3/run_output/hallucination.json 2>&1
#python -m torch.distributed.launch --nproc_per_node=$gpu_nums hallucination.py \
#--wrong_fact_dir datasets/wrong_fact_dataset/temporal/train \
#--neurons_result_dir temporal_res/1118_3 \
#--model_name gpt2 \
#--threshold 0.002 \
#--threshold_filter_DN 0.7 \
#> temporal_res/1118_3/run_output/hallucination.json 2>&1
#python -m torch.distributed.launch --nproc_per_node=$gpu_nums hallucination.py \
#--wrong_fact_dir datasets/wrong_fact_dataset/temporal/train \
#--neurons_result_dir temporal_res/1118_3 \
#--model_name gpt2 \
#--threshold 0.002 \
#--threshold_filter_DN 0.5 \
#> temporal_res/1118_3/run_output/hallucination.json 2>&1
#python -m torch.distributed.launch --nproc_per_node=$gpu_nums hallucination.py \
#--wrong_fact_dir datasets/wrong_fact_dataset/temporal/train \
#--neurons_result_dir temporal_res/1118_3 \
#--model_name gpt2 \
#--threshold 0.002 \
#--threshold_filter_DN 0.3 \
#> temporal_res/1118_3/run_output/hallucination.json 2>&1


#python -m torch.distributed.launch --nproc_per_node=$gpu_nums hallucination.py \
#--wrong_fact_dir datasets/wrong_fact_dataset/temporal/train \
#--neurons_result_dir temporal_res/1118_3 \
#--model_name gpt2 \
#--threshold 0.000001 \
#--threshold_filter_DN 0.7 \
#> temporal_res/1118_3/run_output/hallucination.json 2>&1
#python -m torch.distributed.launch --nproc_per_node=$gpu_nums hallucination.py \
#--wrong_fact_dir datasets/wrong_fact_dataset/temporal/train \
#--neurons_result_dir temporal_res/1118_3 \
#--model_name gpt2 \
#--threshold 0.00001 \
#--threshold_filter_DN 0.7 \
#> temporal_res/1118_3/run_output/hallucination.json 2>&1
#python -m torch.distributed.launch --nproc_per_node=$gpu_nums hallucination.py \
#--wrong_fact_dir datasets/wrong_fact_dataset/temporal/train \
#--neurons_result_dir temporal_res/1118_3 \
#--model_name gpt2 \
#--threshold 0.000005 \
#--threshold_filter_DN 0.7 \
#> temporal_res/1118_3/run_output/hallucination.json 2>&1
#python -m torch.distributed.launch --nproc_per_node=$gpu_nums hallucination.py \
#--wrong_fact_dir datasets/wrong_fact_dataset/temporal/train \
#--neurons_result_dir temporal_res/1118_3 \
#--model_name gpt2 \
#--threshold 0.000005 \
#--threshold_filter_DN 0.5 \
#> temporal_res/1118_3/run_output/hallucination.json 2>&1
# 原始的设置:
#python hallucination.py \
#--wrong_fact_dir datasets/wrong_fact_dataset/temporal/train \
#--neurons_result_dir temporal_res/1118_3 \
#--model_name gpt2 \
#--threshold 0.0000005 \
#--threshold_filter_DKN 0.5 \
#> temporal_res/1118_3/run_output/hallucination_5.json 2>&1
# 设置神经元数目相同的设置：
#python -m torch.distributed.launch --nproc_per_node=$gpu_nums tmp_hallucination.py \
#--wrong_fact_dir datasets/wrong_fact_dataset/temporal/train \
#--neurons_result_dir temporal_res/1118_3 \
#--model_name gpt2 \
#--threshold 0.0000005 \
#--threshold_filter_DKN 0.7 \
#> temporal_res/1118_3/run_output/hallucination_2.json 2>&1
