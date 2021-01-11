import argparse
import cv2
import time
from pathlib import Path
import numpy as np
from numpy import random
import torch
import torch.backends.cudnn as cudnn

from utils.utils import ImageReader, VideoReader
from utils.torch_utils import select_device

def get_args():
    parser = argparse.ArgumentParser('Object-pose detector.')

    # Common arguments
    parser.add_argument('--tasks', nargs='+', type=str, default=['Object', 'Pose'], help='detecting tasks to implement')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    # Object detection arguments
    parser.add_argument('--object-weights', nargs='+', type=str, default='weights/yolov3.pt', help='object detection model.pt path(s)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (height, weight)')

    # Pose detection arguments
    parser.add_argument('--pose-weights', type=str, default='weights/checkpoint_iter_370000.pth', help='pose detection model path')
    parser.add_argument('--track', type=int, default=1, help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')

    args = parser.parse_args()
    print(args)

    return args


def detect():
    args = get_args()
    source = args.source
    
    # Set Dataloader
    is_webcam = source.isnumeric() or source.lower().startswith(('rtsp://', 'rtmp://', 'http://'))
    view_img = False
    save_img = False
    frame_provider = ImageReader(source)
    if is_webcam: 
        # Webcam
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        frame_provider = VideoReader(source)
    else:
        # Images
        args.track = 0
        save_img = True

    # Initialize
    device = select_device(args.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    tasks = args.tasks
    if 'Object' in tasks:
        from models.experimental import attempt_load
        from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
            strip_optimizer, set_logging, increment_path
        from utils.plots import plot_one_box
        from utils.torch_utils import load_classifier, time_synchronized
        from utils.datasets import letterbox
    
        # Load model
        object_model = attempt_load(args.object_weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(args.img_size, s=object_model.stride.max())  # check img_size
        if half:
            object_model.half()  # to FP16

        # Get names and colors
        object_names = object_model.module.names if hasattr(object_model, 'module') else object_model.names
        object_colors = [[random.randint(0, 255) for _ in range(3)] for _ in object_names]

        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = object_model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    
    if 'Pose' in tasks:
        from models.with_mobilenet import PoseEstimationWithMobileNet
        from modules.keypoints import extract_keypoints, group_keypoints
        from modules.load_state import load_state
        from modules.pose import Pose, track_poses
        from pose_detector import infer_fast

        pose_model = PoseEstimationWithMobileNet()
        checkpoint = torch.load(args.pose_weights, map_location='cpu')
        load_state(pose_model, checkpoint)

        pose_model = pose_model.eval()
        if device.type != 'cpu':
            pose_model = pose_model.cuda()

        stride = 8
        upsample_ratio = 4
        num_keypoints = Pose.num_kpts
        previous_poses = []

    for img in frame_provider:
        total_tic = time.time()
        orig_img = img.copy()
        current_poses = []
        if 'Pose' in tasks:
            heatmaps, pafs, scale, pad = infer_fast(pose_model, 
                                                    img, 
                                                    args.height_size, 
                                                    stride, 
                                                    upsample_ratio, 
                                                    device.type=='cpu')

            total_keypoints_num = 0
            all_keypoints_by_type = []
            for kpt_idx in range(num_keypoints):  # 19th for bg
                total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

            pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
            for kpt_id in range(all_keypoints.shape[0]):
                all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
                all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
            
            for n in range(len(pose_entries)):
                if len(pose_entries[n]) == 0:
                    continue
                pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
                for kpt_id in range(num_keypoints):
                    if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                        pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                        pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
                pose = Pose(pose_keypoints, pose_entries[n][18])
                current_poses.append(pose)

            if args.track:
                track_poses(previous_poses, current_poses, smooth=args.smooth)
                previous_poses = current_poses
            
        det = None
        if 'Object' in tasks:
            img = orig_img.copy()

            # Padded resize
            img = letterbox(orig_img, new_shape=imgsz)[0]

            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = object_model(img, augment=args.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, classes=args.classes, agnostic=args.agnostic_nms)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orig_img.shape).round()

        # Draw pose detection results
        for pose in current_poses:
            pose.draw(orig_img)
            for pose in current_poses:
                cv2.rectangle(orig_img, (pose.bbox[0], pose.bbox[1]),
                            (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
                if args.track:
                    cv2.putText(orig_img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))

        # Draw object detection results
        if det is not None:
            # Write results
            for *xyxy, conf, cls in reversed(det):
                if save_img or view_img:  # Add bbox to image
                    label = '%s %.2f' % (object_names[int(cls)], conf)
                    plot_one_box(xyxy, orig_img, label=label, color=object_colors[int(cls)], line_thickness=3)

        # Stream results
        if view_img:
            total_toc = time.time()
            total_time = total_toc - total_tic
            frame_rate = 1 / total_time
            print('{:.2f} fps'.format(frame_rate))
            cv2.imshow('Results', orig_img)
            key = cv2.waitKey(1)
            if key == 27:  # esc
                return
    

if __name__ == '__main__':
    detect()


