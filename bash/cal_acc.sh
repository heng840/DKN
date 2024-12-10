export CUDA_VISIBLE_DEVICES=0,1

#python -m torch.distributed.launch --nproc_per_node=$gpu_nums cal_acc_for_baseline.py \
python cal_acc_for_baseline.py \
--baseline_result_first_dir /home/chenyuheng/KN2/kn2/temporal_res/llama7b_1226/res_wo_acc/1 \
--split_rank 0 \
> temporal_res/llama7b_1226/xshell_output/0_output_for_with_acc.json 2>&1
#--model_name gpt2 \
#--dataset_json Templama/train.jsonl \
#--neurons_result_dir temporal_res/1205/all_baseline \
#--baseline_result_first_dir temporal_res/1205/main_temporal_res \