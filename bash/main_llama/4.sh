export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9
if [ ! -d "temporal_res/llama7b_1226/xshell_output" ]; then
    mkdir "temporal_res/llama7b_1226/xshell_output"
fi
python main.py \
--neurons_result_dir temporal_res/llama7b/main_res/part4 \
--dataset_json /home/chenyuheng/KN2/kn2/Templama/train_split/part_4.jsonl \
> temporal_res/llama7b_1226/xshell_output/4.json 2>&1

