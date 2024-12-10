if [ ! -d "temporal_res/llama7b_1226/xshell_output" ]; then
    mkdir "temporal_res/llama7b_1226/xshell_output"
fi
#export CUDA_VISIBLE_DEVICES=6,7,8,9,5
#python main.py \
#--neurons_result_dir temporal_res/llama7b_1226/main_res/part3 \
#--dataset_json /home/chenyuheng/KN2/kn2/Templama/train_split/part_3.jsonl \
#> temporal_res/llama7b_1226/xshell_output/3.json 2>&1

export CUDA_VISIBLE_DEVICES=6,7,8,9,5
python main.py \
--neurons_result_dir temporal_res/llama7b_1226/main_res/part1 \
--dataset_json /home/chenyuheng/KN2/kn2/Templama/train_split/part_1.jsonl \
> temporal_res/llama7b_1226/xshell_output/3.json 2>&1