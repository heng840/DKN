export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9


python hallucination.py \
--model_name /home/chenyuheng/KN2/Llama/Llama7bChat \
--use_complex_text \
--data_file datasets/wrong_fact_dataset/temporal/train-llama-complex/lama.jsonl \
--wrong_fact_dir datasets/wrong_fact_dataset/temporal/train-llama-complex \
--neurons_result_dir temporal_res/llama7b_1226/res_wo_acc \
--hallucination_parent_dir Hallucination_res/llama/USE_complex_text_GOLDEN \
--run_mode not_split_check \
> /home/chenyuheng/KN2/kn2/temporal_res/llama7b_1226/xshell_output/hallucination_USE_complex_text_GOLDEN.json 2>&1