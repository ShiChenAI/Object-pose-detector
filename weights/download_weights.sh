#!/bin/bash
# Download latest models from https://github.com/ultralytics/yolov3/releases
# Usage:
#    $ bash weights/download_weights.sh

#python - <<EOF
#from utils.google_utils import attempt_download

#for x in ['yolov3', 'yolov3-spp', 'yolov3-tiny']:
#    attempt_download(f'/weights/{x}.pt')

#EOF

for model_name in 'yolov3.pt' 'yolov3-tiny.pt' 'yolov3-spp.pt'; do
    if [ ! -f 'weights/${model_name}' ]; then
        echo 'Downloading https://github.com/ultralytics/yolov3/releases/download/v9.0/${model_name}'
        wget -P weights/ https://github.com/ultralytics/yolov3/releases/download/v9.0/${model_name}
    fi
done

if [ ! -f 'weights/checkpoint_iter_370000.pth' ]; then
    echo 'Downloading https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth'
    wget -P weights/ https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth
fi
