import argparse
import os
import cv2
import time
from pathlib import Path
import numpy as np
from numpy import random
import torch
import torch.backends.cudnn as cudnn

from utils.utils import ImageReader, VideoReader, postprocess, display_results
from utils.torch_utils import select_device
from utils.general import increment_path
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import load_classifier, time_synchronized
from utils.datasets import letterbox
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from pose_detector import normalize, pad_width

class ImageProcessor:
    def __init__(self, params):
        self.__params = params

    def _pose_dect_init(self, device):
        """Initialize the pose detection model.

        Arguments:
            device {torch.device}: device to implement the models on.

        Returns:
            PoseEstimationWithMobileNet: initialized OpenPose model.
        """        

        weight_path = self.__params.pose_weights
        model = PoseEstimationWithMobileNet()
        weight = torch.load(weight_path, map_location='cpu')
        load_state(model, weight)
        model = model.eval()
        if device.type != 'cpu':
            model = model.cuda()

        return model

    def _infer_fast(self, **kwargs):
        """Pose inference using fast OpenPose model.
        
        Arguments:
            img {ndarray}: input image.
            model {PoseEstimationWithMobileNet: initialized OpenPose model.
            pad_value {tuple}: pad value.
            img_mean {tuple}: mean image value.
            img_scale {float}: scale image value.

        Returns:
            ndarray: heatmaps.
            ndarray: pafs.
            float: scale.
            list: pad.
        """        

        img = kwargs.get('img', None)
        model = kwargs.get('model', None)
        pad_value = kwargs.get('pad_value', (0, 0, 0))
        img_mean = kwargs.get('img_mean', (128, 128, 128))
        img_scale = kwargs.get('img_scale', 1/256)
        use_cuda = kwargs.get('use_cuda', False)
        
        stride = self.__params.stride
        upsample_ratio = self.__params.upsample_ratio
        height_size = self.__params.height_size

        height, _, _ = img.shape
        scale = height_size / height

        scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        scaled_img = normalize(scaled_img, img_mean, img_scale)
        min_dims = [height_size, max(scaled_img.shape[1], height_size)]
        padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
        if use_cuda:
            tensor_img = tensor_img.cuda()

        stages_output = model(tensor_img)

        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        return heatmaps, pafs, scale, pad
        
    def _dect_pose(self, **kwargs):
        """Detect poses.
        Arguments:
            img {ndarray}: input image.
            model {PoseEstimationWithMobileNet}: initialized OpenPose model.
            previous_poses {list}: previous poses for tracking mode.
        Returns:
            list: detected poses.
        """        

        img = kwargs.get('img', None)
        model = kwargs.get('model', None)
        previous_poses = kwargs.get('previous_poses', None)
        use_cuda = kwargs.get('use_cuda', False)
        track = self.__params.track
        smooth = self.__params.smooth
        stride = self.__params.stride
        upsample_ratio = self.__params.upsample_ratio
        num_keypoints = Pose.num_kpts
        
        #orig_img = img.copy()
        heatmaps, pafs, scale, pad = self._infer_fast(model=model, 
                                                      img=img,
                                                      use_cuda=use_cuda)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        
        current_poses = []
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

        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses

        return current_poses

    def _object_dect_init(self, device):
        """Initialize object detection model.

        Arguments:
            device {torch.device}: device to implement the models on.

        Returns:
            Ensemble: initialized YOLOv3 model.
            list: object names.
            list: object colors.
        """   

        weight_path = self.__params.object_weights
        input_size = self.__params.img_size
        
        # Load object detection model
        object_model = attempt_load(weight_path, map_location=device)  # load FP32 model
        imgsz = check_img_size(input_size, s=object_model.stride.max())  # check img_size
        if device.type != 'cpu':
            object_model.half()  # to FP16

        # Get object names and colors
        object_names = object_model.module.names if hasattr(object_model, 'module') else object_model.names
        object_colors = [[random.randint(0, 255) for _ in range(3)] for _ in object_names]

        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = object_model(img.half() if device.type != 'cpu' else img) if device.type != 'cpu' else None  # run once

        return object_model, object_names, object_colors

    def _detect_object(self, **kwargs):
        """Detect objects.

        Arguments:
            img {ndarray}: input image.
            device {torch.device}: device to implement the models on.
            model {Ensemble}: initialized YOLOv3 model.

        Returns:
            dictionary: detected objects.
        """      

        framed_img = kwargs.get('img', None)
        device = kwargs.get('device', None)
        model = kwargs.get('model', None)
        input_size = self.__params.img_size
        conf_thres = self.__params.conf_thres
        iou_thres = self.__params.iou_thres

        img = framed_img.copy()

        # Padded resize
        img = letterbox(framed_img, new_shape=input_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if device.type != 'cpu' else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():
            pred = model(img, augment=False)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres)

            # Process detected objects
            out = postprocess(pred, img.shape, framed_img.shape)

        return out

    def init_models(self, device):
        """Initialize object/pose detection models.

        Arguments:
            device {torch.device}: device to implement the models on.

        Returns:
            PoseEstimationWithMobileNet: initialized OpenPose model.
            Ensemble: initialized YOLOv3 model.
            list: object names.
            list: object colors.
        """        

        pose_model = self._pose_dect_init(device)
        object_model, object_names, object_colors = self._object_dect_init(device)
        
        return pose_model, object_model, object_names, object_colors

    def process_frame(self, **kwargs):
        """Process a single frame.

        Arguments:
            frame {ndarray}: frame image.
            device {torch.device}: device to implement the models on.

        Returns:
            dictionary: detected objects.
            list: detected poses.
        """   

        frame = kwargs.get('frame', None)
        device = kwargs.get('device', None)
        
        # Detect objects
        object_model = kwargs.get('object_model', None)
        out = self._detect_object(img=frame,
                                  model=object_model,
                                  device=device) 

        # Detect poses
        pose_model = kwargs.get('pose_model', None)
        previous_poses = kwargs.get('previous_poses', None)
        current_poses = self._dect_pose(img=frame,
                                        model=pose_model,
                                        previous_poses=previous_poses, 
                                        use_cuda=device.type != 'cpu')

        return out, current_poses

    def process(self, **kwargs):
        """Process images/video.
        
        Arguments:
            images {string}: paths of the input image.
            video_src {string}: path pf the input video (0 for webcam).
            show_results {boolen}: whether to show the results or not.

        Returns:
            dictionary: detected pose/objects.
        """      
          
        images = kwargs.get('images', '')
        video_src = kwargs.get('video_src', '')
        show_results = kwargs.get('show_results', False)
        device = select_device(self.__params.device)

        # Set frame provider
        frame_provider = ImageReader(images)
        if video_src != '':
            frame_provider = VideoReader(video_src)

        # Initialize object/pose detection models
        pose_model, object_model, object_names, object_colors = self.init_models(device)

        # Process frames
        previous_poses = []
        dets = {}
        frame_idx = 1
        for frame in frame_provider:
            total_tic = time.time()
            out, current_poses = self.process_frame(frame=frame,
                                                    device=device,
                                                    object_model=object_model,
                                                    pose_model=pose_model,
                                                    previous_poses=previous_poses)
            dets[frame_idx] = {'Pose': current_poses, 'Object': out}
            frame_idx += 1
            
            # Show results
            if show_results:
                img_show = display_results(pred=out, 
                                           img=frame, 
                                           obj_list=object_names, 
                                           colors=object_colors, 
                                           current_poses=current_poses,
                                           track=self.__params.track)

                cv2.imshow('Results', img_show)
                total_toc = time.time()
                total_time = total_toc - total_tic
                frame_rate = 1 / total_time
                print('Frame rate:', frame_rate)
                k = cv2.waitKey(1) if video_src != '' else cv2.waitKey(0)
                if k == ord('q'): 
                    if video_src != '':
                        frame_provider.cap.release()
                    cv2.destroyAllWindows()
                    break

        return dets