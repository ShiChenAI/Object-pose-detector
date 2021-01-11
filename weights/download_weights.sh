#!/bin/bash
# Download latest models from https://github.com/ultralytics/yolov3/releases
# Usage:
#    $ bash weights/download_weights.sh

python - <<EOF
from utils.google_utils import attempt_download

for x in ['yolov3', 'yolov3-spp', 'yolov3-tiny']:
    attempt_download(f'{x}.pt')

EOF

wget https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth
