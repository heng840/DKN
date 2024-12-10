export CUDA_VISIBLE_DEVICES=9,8,7,2,0,1,3,4,5,6
# for enhance enhance后来修改了评价指标。所以这个方法弃用了。
#python disturb_query.py --method replace \
#--model_name gpt2 \
#--save_dir Disturb_or_Enhance/enhance_res \
#--run_experiment
#python disturb_query.py --method delete \
#--model_name gpt2 \
#--save_dir Disturb_or_Enhance/enhance_res \
#--run_experiment
#python disturb_query.py --method add \
#--model_name gpt2 \
#--save_dir Disturb_or_Enhance/enhance_res \
#--run_experiment
##
#


## for suppress
#python disturb_query.py --method replace \
#--model_name gpt2 \
#--save_dir Disturb_or_Enhance/suppress_res \
#--run_experiment \
#--mode suppress \
#--change_weight_value 0 \
##
#
#python disturb_query.py --method delete \
#--model_name gpt2 \
#--save_dir Disturb_or_Enhance/suppress_res \
#--run_experiment \
#--mode suppress \
#--change_weight_value 0
#
#
#python disturb_query.py --method add \
#--model_name gpt2 \
#--save_dir Disturb_or_Enhance/suppress_res \
#--run_experiment \
#--mode suppress \
#--change_weight_value 0 \


# for suppress：Llama
python disturb_query.py --method replace \
--model_name /home/chenyuheng/KN2/Llama/Llama7bChat \
--save_dir Disturb_or_Enhance/llama/suppress_res \
--run_experiment \
--mode suppress \
--change_weight_value 0 \
--neurons_result_dir temporal_res/llama7b_1226/res_wo_acc
#

python disturb_query.py --method delete \
--model_name /home/chenyuheng/KN2/Llama/Llama7bChat \
--save_dir Disturb_or_Enhance/llama/suppress_res \
--run_experiment \
--mode suppress \
--change_weight_value 0 \
--neurons_result_dir temporal_res/llama7b_1226/res_wo_acc


python disturb_query.py --method add \
--model_name /home/chenyuheng/KN2/Llama/Llama7bChat \
--save_dir Disturb_or_Enhance/llama/suppress_res \
--run_experiment \
--mode suppress \
--change_weight_value 0 \
--neurons_result_dir temporal_res/llama7b_1226/res_wo_acc
