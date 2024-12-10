
export CUDA_VISIBLE_DEVICES=3
#python cal_fintune_acc.py \
#--model_name /home/chenyuheng/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10
python plot_dkn_correct_or_not.py \
--model_name /home/chenyuheng/KN2/kn2/saved_models/epoch100/model_direct