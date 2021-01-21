import colorsys
import time
import torch
import cv2
import numpy as np
from torch.backends import cudnn
from utils.utils import ImageReader, VideoReader, postprocess, display_results, get_rect_center, get_obj_pose_dist, get_thres, Params
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from pose_detector import normalize, pad_width
from utils.torch_utils import select_device
from utils.datasets import letterbox
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging

class ImageProcessor:
    def __init__(self, project_file):
        self.__params = Params(project_file)
        self._generate_colors()

    def get_params(self):
        return self.__params

    def _generate_colors(self):
        obj_list = self.__params.obj_list
        hsv_tuples = [(x / len(obj_list), 1., 1.)
                      for x in range(len(obj_list))]
        self.obj_colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.obj_colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.obj_colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.obj_colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.


    def _pose_dect_init(self):
        """Initialize the pose detection model.
        Returns:
            PoseEstimationWithMobileNet: initialized OpenPose model.
        """        

        use_cuda = self.__params.use_cuda
        weight_path = self.__params.pose_weight_path
        model = PoseEstimationWithMobileNet()
        weight = torch.load(weight_path, map_location='cpu')
        load_state(model, weight)
        model = model.eval()
        if use_cuda:
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
            stride {integer}: stride value.
            upsample_ratio {integer}: upsample ratio value.
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
        stride = kwargs.get('stride', 8)
        upsample_ratio = kwargs.get('upsample_ratio', 4)
        height_size = self.__params.height_size
        use_cuda = self.__params.use_cuda

        height, width, _ = img.shape
        scale = height_size / height

        scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
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
            model {PoseEstimationWithMobileNet: initialized OpenPose model.
            previous_poses {list}: previous poses for tracking mode.
        Returns:
            list: detected poses.
        """        

        img = kwargs.get('img', None)
        model = kwargs.get('model', None)
        previous_poses = kwargs.get('previous_poses', None)
        track = self.__params.track
        smooth = self.__params.smooth
        stride = self.__params.stride
        upsample_ratio = self.__params.upsample_ratio
        num_keypoints = Pose.num_kpts
        
        #orig_img = img.copy()
        heatmaps, pafs, scale, pad = self._infer_fast(img=img,
                                                      model=model,
                                                      stride=stride,
                                                      upsample_ratio=upsample_ratio)

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
            not_found_num = 0
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
                else:
                    not_found_num += 1
            
            if not_found_num < 11:
                pose = Pose(pose_keypoints, pose_entries[n][18])
                current_poses.append(pose)

        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses

        return current_poses

    def _object_dect_init(self, device):
        """Initialize the object detection model.
        Returns:
            EfficientDetBackbone: initialized EfficientDet model.
            BBoxTransform: bounding box transformer.
            clipBoxes: bounding box cliper.
            integer: input size.
        """        

        weight_path = self.__params.object_weight_path
        input_size = self.__params.input_size        

        model = attempt_load(weight_path, map_location=device)  # load FP32 model
        input_size = check_img_size(input_size, s=model.stride.max())  # check img_size
        model.to(device).eval()

        return model, input_size

    def _detect_object(self, **kwargs):
        """Detect the objects.
        Arguments:
            framed_imgs {list}: input images.
            framed_metas {list}: input metas.
            model {EfficientDetBackbone}: initialized EfficientDet model.
            regressBoxes {BBoxTransform}: bounding box transformer.
            clipBoxes {clipBoxes}: bounding box cliper.
        Returns:
            list: detection results.
        """       
        
        
        conf_threshold = self.__params.conf_threshold
        iou_threshold = self.__params.iou_threshold
        model = kwargs.get('model', None)
        framed_imgs = kwargs.get('framed_imgs', None)
        input_size = kwargs.get('input_size', None)
        device = kwargs.get('device', None)
        img0 = framed_imgs[0].copy()
        # Padded resize
        img = letterbox(img0, new_shape=input_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():
            pred = model(img, augment=False)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_threshold, iou_threshold)

            out = postprocess(pred, img.shape, img0.shape)
        
        return out


    def _associate_objects(self, objs, poses):
        """Associate detected objects with poses.
        Args:
            objs (list): detected objects.
            poses (list): detected poses.
        Returns:
            list: associated objects poses pairs. (pose index, object index)
        """        

        if objs is None or poses is None or len(objs['rois']) == 0 or len(poses) == 0:
            return None, None
        adj_m = np.zeros((len(objs['rois']), len(poses)))
        for i in range(len(objs['rois'])):
            (x1, y1, x2, y2) = objs['rois'][i].astype(np.int)
            obj = self.__params.obj_list[int(objs['class_ids'][i])]
            x_cet, y_cet = get_rect_center(x1, y1, x2, y2)

            for j, pose in enumerate(poses):
                dist = get_obj_pose_dist(obj, x_cet, y_cet, pose)
                adj_m[i, j] = dist

        #print(adj_m)
        #print(adj_m.argmin(axis=1))

        return adj_m, [(pose_idx, obj_idx) for obj_idx, pose_idx in enumerate(adj_m.argmin(axis=1))]

    def _identify_relationship(self, objs, poses, gamma):
        """Identify the relationship between poses and objs
        Args:
            objs (list): detected objects.
            poses (list): detected poses.
            gamma (list): scaling coefficient to strike the relationship analysis for different object.
        Returns:
            ndarray: relationships between poses and objs
        """        

        adj_m, pairs = self._associate_objects(objs, poses)
        if pairs is None:
            return None

        relationships = np.full_like(adj_m, False, dtype=bool)
        for pair in pairs:
            pose = poses[pair[0]]
            thres = get_thres(pose, gamma)
            obj = self.__params.obj_list[objs['class_ids'][pair[1]]]
            relationships[pair[1], pair[0]] = adj_m[pair[1], pair[0]] < thres[obj]

        return relationships

    def _create_scene_graph(self, objs, poses, gamma):
        """Create scene graph from detected poses and objs.
        Args:
            objs (list): detected objects.
            poses (list): detected poses.
            gamma (list): scaling coefficient to strike the relationship analysis for different object.
        Returns:
            dictionary: scene garaph (key-person index, value-scene graph of the person)
        """ 

        scene_graph = {}
        adj_m, pairs = self._associate_objects(objs, poses)
        for i, pose in enumerate(poses):
            person_vertice = 'person'
            if i not in scene_graph:
                scene_graph[i] = {'vertices': [], 'edges': []}
                scene_graph[i]['vertices'].append(person_vertice)
            if pairs is None:
                continue
            for pair in pairs:
                if i != pair[0]:
                    continue

                thres = get_thres(pose, gamma)
                obj = self.__params.obj_list[int(objs['class_ids'][pair[1]])]
                if obj not in self.__params.target_list:
                    continue
                if adj_m[pair[1], pair[0]] > thres[obj]:
                    continue
                if obj not in scene_graph[pair[0]]['vertices']:
                    scene_graph[pair[0]]['vertices'].append(obj)
                edge = (person_vertice, obj, 'wear')
                if edge not in scene_graph[pair[0]]['edges']:
                    scene_graph[pair[0]]['edges'].append(edge)

        return scene_graph

    def init_models(self, device):
        pose_model = self._pose_dect_init()
        object_model, input_size = self._object_dect_init(device)
        return object_model, input_size, pose_model

    def process_frame(self, **kwargs):
        frame = kwargs.get('frame', None)
        input_size = kwargs.get('input_size', None)
        object_model = kwargs.get('object_model', None)
        pose_model = kwargs.get('pose_model', None)
        previous_poses = kwargs.get('previous_poses', None)
        device = kwargs.get('device', None)
        out = self._detect_object(framed_imgs=[frame],
                                    model=object_model,
                                    input_size=input_size,
                                    device=device)
        ori_imgs = [frame]
        current_poses = self._dect_pose(img=frame,
                                        model=pose_model,
                                        previous_poses=previous_poses)
        
        gamma = {}
        for i, target in enumerate(self.__params.target_list):
            gamma[target] = self.__params.gamma[i]

        scene_graph = self._create_scene_graph(out, current_poses, gamma=gamma)
        #scene_graph = self._create_scene_graph(out, current_poses, gamma={'helmet': 0.25, 'visor': 0.2, 'mask': 0.2, 'safetybelt': 0.2})

        return out, current_poses, ori_imgs, scene_graph

    def process(self, **kwargs):
        """Process images.
        
        Arguments:
            images {string}: paths pf the input image.
            video_src {string}: path pf the input video (0 for webcam).
            show_results {boolen}: whether to show the results or not.
        """      
          
        images = kwargs.get('images', '')
        video_src = kwargs.get('video_src', '')
        show_results = kwargs.get('show_results', False)
        device = kwargs.get('device', None)
        frame_provider = ImageReader(images)
        if video_src != '':
            frame_provider = VideoReader(video_src)
        
        object_model, input_size, pose_model = self.init_models(device)

        previous_poses = []
        for frame in frame_provider:
            total_tic = time.time()
            out, current_poses, ori_imgs, scene_graph = self.process_frame(frame=frame, 
                                                                            input_size=input_size,
                                                                            object_model=object_model,
                                                                            pose_model=pose_model,
                                                                            previous_poses=previous_poses,
                                                                            device=device)
                
            img_show = display_results(pred=out, 
                                       img=ori_imgs[0], 
                                       obj_list=self.__params.obj_list, 
                                       colors=self.obj_colors, 
                                       current_poses=current_poses,
                                       track=self.__params.track)
            if show_results:
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

        return img_show, scene_graph