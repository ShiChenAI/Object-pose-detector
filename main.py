import argparse
import os
from pathlib import Path

from utils.general import increment_path
from processor import ImageProcessor

def get_args():
    parser = argparse.ArgumentParser('Object-pose detector.')

    # Common arguments
    parser.add_argument('--project', default='projects/coco', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--tasks', nargs='+', type=str, default=['Object', 'Pose'], help='detecting tasks to implement')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

    # Object detection arguments
    parser.add_argument('--object-weights', nargs='+', type=str, default='weights/yolov3.pt', help='object detection model.pt path(s)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (height, weight)')

    # Pose detection arguments
    parser.add_argument('--pose-weights', type=str, default='weights/checkpoint_iter_370000.pth', help='pose detection model path')
    parser.add_argument('--track', type=int, default=1, help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--upsample_ratio', type=int, default=4)
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')

    args = parser.parse_args()
    print(args)

    return args

if __name__ == '__main__':
    args = get_args()
    ip = ImageProcessor(args)
    dets = ip.process(video_src=0, show_results=True)

    # Directories
    save_dir = Path(increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok))  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    txt_path = os.path.join(save_dir, 'results.txt')

    # Write results
    with open(txt_path, 'a') as f:
        for frame_idx, det in dets.items():
            f.write('Frame {}\n'.format(frame_idx))
            for pose in det['Pose']:
                f.write(str(pose.id) + '\n')
                f.write(str(pose.keypoints) + '\n')

            for k, v in det['Object'].items():
                f.write(str(k) + '\n')
                f.write(str(v) + '\n')