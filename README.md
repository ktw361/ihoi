# Fitting epic_clip_v3

CUDA_VISIBLE_DEVICES=0 python -m pudb temporal/fit_mvho.py  dataset.image_sets=/home/skynet/Zhifan/epic_analysis/hos/tools/model-input-Feb03.json debug_index=1 optim_multiview.num_inits=40