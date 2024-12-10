#export HF_HOME=/netcache/huggingface/
export CUDA_VISIBLE_DEVICES=8,9,7,6,5,4,3,2,1
python filter_test_data.py \
#--test_dataset /home/chenyuheng/KN2/kn2/Templama/filtered_test.jsonl \
#--filtered_test_dataset /home/chenyuheng/KN2/kn2/Templama/filtered_test_dkn_can_answer_and_kn_cannot.jsonl