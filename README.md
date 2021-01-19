# Object-pose-detector

## Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Reference](#reference)

## Requirements
0. Python 3.x
1. torch>=1.7.0
2. torchvision>=0.8.1
3. Tensorboard>=2.2
4. matplotlib>=3.2.2
5. numpy>=1.18.5
6. opencv-python>=4.1.2
7. Pillow
8. PyYAML>=5.3
9. scipy>=1.4.1
10. tqdm>=4.41.0
11. pycocotools==2.0

## Installation
1. Clone this repository.
```
git clone git@github.com:ShiChenAI/Object-pose-detector.git
cd Object-pose-detector
```

2. Install the dependencies. The code should run with PyTorch 1.7.0.
```
pip install -r requirements.txt 
```

3. Download pretrained weights from https://github.com/ultralytics/yolov3/releases and https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch
```
sh weights/download_weights.sh
```

## Usage
1. Create dataset.yaml
.yaml is the dataset configuration file that defines 1) a path to a directory of training images (or path to a *.txt file with a list of training images), 2) a path to a directory of validation images, 3) the number of classes, 4) a list of class names:
```
# train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]
train: ../coco2017/train2017/
val: ../coco2017/train2017/

# number of classes
nc: 80

# class names
names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 
        'teddy bear', 'hair drier', 'toothbrush']
```

2. Organize directories
Organize the custom train and val images and labels according to the example below. For example:
```
    datasets/
        -coco2017/
            -train2017/
                -000000000001.jpg
                -000000000001.txt
                -000000000002.jpg
                -000000000002.txt
                -000000000003.jpg
                -000000000003.txt
            -val2017/
                -000000000004.jpg
                -000000000004.txt
                -000000000005.jpg
                -000000000005.txt
                -000000000006.jpg
                -000000000006.txt
```
If you already have a custom dataset in COCO format or VOC format, you can use the scripts `/datasets/voc2coco.py` and `/datasets/coco2yolo.py` to convert it to YOLOv3 format.

3. Train
Train a YOLOv3 model on the custom data by specifying dataset, batch-size, image size and pretrained weight `--weights yolov3.pt`.
```
# Train YOLOv3 on COCO128 for 5 epochs
python train_yolov3.py --img 640 --batch 16 --epochs 5 --data data/coco128.yaml --weights weights/yolov3.pt
```

4. Inference
`main.py` runs inference on a variety of sources and saving results to `--project/--name/results.txt` by specifying `--tasks` for object detection and/or pose detection.
```
python main.py --source 0 --tasks Object Pose --device 0
```

## Reference
 * [YOLOv3](https://github.com/ultralytics/yolov3)
 * [Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch)
