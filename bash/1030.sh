#export CUDA_VISIBLE_DEVICES=3
#python freeze_finetune.py
#
#export CUDA_VISIBLE_DEVICES=3
#python main.py
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9
if [ ! -d "temporal_res/1205" ]; then
    mkdir "temporal_res/1205"
fi
gpu_nums=10
#python -m torch.distributed.launch --nproc_per_node=$gpu_nums main.py \
#--model_name gpt2 \
#--dataset_json Templama/train.jsonl \
#--neurons_result_dir temporal_res/1205 \
#> temporal_res/1205/output.json 2>&1

# for baseline

python -m torch.distributed.launch --nproc_per_node=$gpu_nums cal_acc_for_baseline.py \
--model_name gpt2 \
--dataset_json Templama/train.jsonl \
--neurons_result_dir temporal_res/1205/all_baseline \
--baseline_result_first_dir temporal_res/1205/main_temporal_res \
> temporal_res/1205/output_for_baseline.json 2>&1


# tmp :for hallucination

python -m torch.distributed.launch --nproc_per_node=$gpu_nums hallucination.py \
--wrong_fact_dir datasets/wrong_fact_dataset/temporal/train \
--neurons_result_dir temporal_res/1118_3 \
--model_name gpt2 \
--threshold 0.0000005 \
--threshold_filter_DKN 0.5 \
> temporal_res/1118_3/run_output/hallucination_5.json 2>&1

python -m torch.distributed.launch --nproc_per_node=$gpu_nums hallucination.py \
--wrong_fact_dir datasets/wrong_fact_dataset/temporal/train \
--neurons_result_dir temporal_res/1118_3 \
--model_name gpt2 \
--threshold 0.0000005 \
--threshold_filter_DKN 0.8 \
> temporal_res/1118_3/run_output/hallucination_8.json 2>&1