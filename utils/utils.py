import cv2
import numpy as np
import yaml
from utils.general import scale_coords

class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img

class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img

def postprocess(pred, pre_shape, ori_shape):
    rois = []
    class_ids = []
    scores = []
    for det in pred:
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(pre_shape[2:], det[:, :4], ori_shape).round()
            for *xyxy, conf, cls_ in det:
                rois.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                class_ids.append(int(cls_))
                scores.append(float(conf))

    out = {'rois': np.array(rois),
           'class_ids': np.array(class_ids),
           'scores': np.array(scores)}
    return out

def display_results(pred, img, obj_list, colors, current_poses, track):
    for pose in current_poses:
        pose.draw(img)
        """
        cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                        (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
        """
        if False:
            cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))

    if len(pred['rois']) == 0:
        return img

    for i in range(len(pred['rois'])):
        (x1, y1, x2, y2) = pred['rois'][i]
        cv2.rectangle(img, (x1, y1), (x2, y2), colors[pred['class_ids'][i]], 2)
        obj = obj_list[pred['class_ids'][i]]
        score = pred['scores'][i]

        cv2.putText(img, '{}, {:.3f}'.format(obj, score),
                    (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 2)

    return img

def get_pos_line_dis(point_x, point_y, line_x1, line_y1, line_x2, line_y2):
    a = line_y2 - line_y1
    b = line_x1 - line_x2
    c = line_x2 * line_y1 - line_x1 * line_y2
    dis = (math.fabs(a * point_x + b * point_y + c)) / (math.pow(a * a + b * b, 0.5))

    return dis


def get_pos_dis_2d(point1_x, point1_y, point2_x, point2_y):
    v1 = np.array([point1_x, point1_y])
    v2 = np.array([point2_x, point2_y])
    dis = np.linalg.norm(v1 - v2)

    return dis


def is_pos_in_rect(point_x, point_y, rect_x1, rect_y1, rect_x2, rect_y2):
    if (point_x > rect_x1) and (point_x < rect_x2) and (point_y > rect_y1) and (point_y < rect_y2):
        return True
    else:
        return False


def get_closest_dist_rect(point_x, point_y, rect_x1, rect_y1, rect_x2, rect_y2):
    if is_in_line_space(point_x, point_y, rect_x1, rect_y1, rect_x2, rect_y2):
        dis1 = get_pos_line_dis(point_x, point_y, rect_x1, rect_y1, rect_x1, rect_y2)
        dis2 = get_pos_line_dis(point_x, point_y, rect_x1, rect_y1, rect_x2, rect_y1)
        dis3 = get_pos_line_dis(point_x, point_y, rect_x2, rect_y1, rect_x2, rect_y2)
        dis4 = get_pos_line_dis(point_x, point_y, rect_x1, rect_y2, rect_x2, rect_y2)
    else:
        dis1 = get_pos_dis_2d(point_x, point_y, rect_x1, rect_y1)
        dis2 = get_pos_dis_2d(point_x, point_y, rect_x1, rect_y2)
        dis3 = get_pos_dis_2d(point_x, point_y, rect_x2, rect_y1)
        dis4 = get_pos_dis_2d(point_x, point_y, rect_x2, rect_y2)

    return min(dis1, dis2, dis3, dis4)


def is_in_line_space(point_x, point_y, line_x1, line_y1, line_x2, line_y2):
    if min(line_x1, line_x2) < point_x < max(line_x1, line_x2) or min(line_y1, line_y2) < point_y < max(line_y1, line_y2):
        return True
    else:
        return False


def is_pos_within_region(point_x, point_y, rect_x1, rect_y1, rect_x2, rect_y2, threshold):
    mid_x, mid_y = get_rect_mid(rect_x1, rect_y1, rect_x2, rect_y2)
    if is_pos_in_rect(point_x, point_y, rect_x1, rect_y1, rect_x2, rect_y2) or \
       get_closest_dist_rect(point_x, point_y, rect_x1, rect_y1, rect_x2, rect_y2) < threshold:
       #math_utils.get_closest_dist_rect(point_x, point_y, rect_x1, rect_y1, rect_x2, rect_y2, threshold) < threshold:
       return True
    else:
        return False


def get_rect_mid(rect_x1, rect_y1, rect_x2, rect_y2):
    return int((rect_x1 + rect_x2) / 2), int((rect_y1 + rect_y2) / 2)


def get_vec_angle(v1, v2):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180/math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180/math.pi)
    # print(angle2)
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle

def get_rect_center(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def is_dist_less_thres(x1, y1, x2, y2, threshold):
    v1 = np.array([x1, y1])
    v2 = np.array([x2, y2])
    r = np.linalg.norm(v1 - v2)
    if np.linalg.norm(v1 - v2) < threshold:
        return True
    else:
        return False

def get_trunk_center(pose):
    x_neck, y_neck = pose.keypoints[1]
    x_r_hip, y_r_hip = pose.keypoints[8]
    x_l_hip, y_l_hip = pose.keypoints[11]

    return int(max(x_neck, x_r_hip, x_l_hip)/2 + min(x_neck, x_r_hip, x_l_hip)/2), int(max(y_neck, y_r_hip, y_l_hip)/2 + min(y_neck, y_r_hip, y_l_hip)/2)

def get_obj_pose_dist(obj, x_cet, y_cet, pose):
    x_nose, y_nose = pose.keypoints[1] #if pose.keypoints[0][0] == -1 else pose.keypoints[0]
    if obj == 'safetybelt':
        x_trunk_cet, y_trunk_cet = get_trunk_center(pose)
        dist = get_pos_dis_2d(x_cet, y_cet, x_trunk_cet, y_trunk_cet)
    else:
        dist = get_pos_dis_2d(x_cet, y_cet, x_nose, y_nose) 

    return dist

def get_thres(pose, gamma={'helmet': 0.2, 'visor': 0.2, 'mask': 0.2, 'safetybelt': 0.2}):
    thres = {}
    x_neck, y_neck = pose.keypoints[0]
    x_l_hip, y_l_hip = pose.keypoints[8]
    x_r_hip, y_r_hip = pose.keypoints[11]
    l_dist = get_pos_dis_2d(x_neck, y_neck, x_l_hip, y_l_hip)
    r_dist = get_pos_dis_2d(x_neck, y_neck, x_r_hip, y_r_hip)
    for k, v in gamma.items():
        thres[k] = v * max(l_dist, r_dist)

    return thres

def get_models_info(model_dir):
    models_info = []
    for root_dir, _, file_names in os.walk(model_dir, topdown=False):
        for file_name in file_names:
            if file_name.endswith('.pth'):
                fn, _ = os.path.splitext(file_name)
                coef, _, _ = fn.split('-')[1].split('_')
                model_path = os.path.join(root_dir, file_name)
                models_info.append((fn, coef, model_path))
    
    return models_info

def update_model_path(cfg_path, coef, model_path):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        result = f.read()
        x = yaml.load(result, Loader=yaml.FullLoader)
        x['object_weight_path'] = model_path
        x['compound_coef'] = int(coef[1])
        with open(cfg_path, 'w', encoding='utf-8') as w_f:
            yaml.dump(x, w_f)

def get_dir_name(path):
    dir_name = path.split('/')[-1]
    if '.' in dir_name:
        dir_name = dir_name.split('.')[0]
    return dir_name

def split_path(path):
    return path.split('/')[-3], path.split('/')[-2], path.split('/')[-1]
    
class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)