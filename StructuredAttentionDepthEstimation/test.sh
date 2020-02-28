#!/bin/bash
clear
echo $"Start testing ..."
model_defition_path="./prototxt/test_SAN.prototxt"
# model_weights="./models/pretrained_model/ResNet-50-pratrained-model.caffemodel"
model_weights="./models/SAN_1_iter_400.caffemodel"
pred_output_save="./output/SAN_results.npy"
kitti_data_root=""
prediction_layer_blob_name="final_output"
gpu_id=0
crop_type_for_evaluation="garg"
python test_kitti_depth.py --model_def=$model_defition_path --weights=$model_weights --pred_file=$pred_output_save --data_root=$kitti_data_root --prediction_blob=$prediction_layer_blob_name --gpu=$gpu_id
echo $"Start evaluating"
python ./utils/evaluation_depth.py --kitti_dir=$kitti_data_root --pred_file=$pred_output_save --which_crop=$crop_type_for_evaluation
