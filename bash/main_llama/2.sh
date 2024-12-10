#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9
#python main.py \
#--neurons_result_dir temporal_res/llama7b/main_res/part1 \
#--dataset_json /home/chenyuheng/KN2/kn2/Templama/train_split/part_1.jsonl

if [ ! -d "temporal_res/llama7b_1226/xshell_output" ]; then
    mkdir "temporal_res/llama7b_1226/xshell_output"
fi
export CUDA_VISIBLE_DEVICES=5,4,3,2,1,0

python main.py \
--neurons_result_dir temporal_res/llama7b_1226/main_res/part2 \
--dataset_json /home/chenyuheng/KN2/kn2/Templama/train_split/part_2.jsonl \
> temporal_res/llama7b_1226/xshell_output/2.json 2>&1