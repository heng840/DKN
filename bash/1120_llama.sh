
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,8,9
#
export CUDA_VISIBLE_DEVICES=1
#gpu_nums=8
#python -m torch.distributed.launch --nproc_per_node=$gpu_nums main.py \

python main.py \
--model_name /home/chenyuheng/KN2/Llama/Llama7bChat \
--dataset_json Templama/train.jsonl \
--neurons_result_dir temporal_res/1118_3 \
--batch_size 1 \
--steps 1 \
> temporal_res/1118_3/run_output/main_llama2.json 2>&1


export HF_HOME=/netcache/huggingface/ CUDA_VISIBLE_DEVICES=0,1,2 python load_llama.py