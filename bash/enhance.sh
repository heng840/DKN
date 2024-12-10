export CUDA_VISIBLE_DEVICES=9,8,7,2,0,1,3,4,5,6
# for enhance
python enhance_dkn4perturbed_query.py \
--neurons_result_dir /home/chenyuheng/KN2/kn2/temporal_res/llama7b_1226/res_wo_acc \
--filtered_test_dataset /home/chenyuheng/KN2/kn2/Templama/llama/for_enhance \
--model_name /home/chenyuheng/KN2/Llama/Llama7bChat \
--save_dir Disturb_or_Enhance/llama/enhance_res
