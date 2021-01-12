import cv2
from utils.general import scale_coords
import numpy as np
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
                rois.append([float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])])
                class_ids.append(cls_.cpu().detach().numpy())
                scores.append(conf.cpu().detach().numpy())
    out = {'rois': np.array(rois),
           'class_ids': np.array(class_ids),
           'scores': np.array(scores)}
    return out

def display_results(pred, img, obj_list, colors, current_poses, track):
    #imgs[i] = cv2.addWeighted(orig_img, 0.6, imgs[i], 0.4, 0)
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
        (x1, y1, x2, y2) = pred['rois'][i].astype(np.int)
        cv2.rectangle(img, (x1, y1), (x2, y2), colors[int(pred['class_ids'][i])], 2)
        obj = obj_list[int(pred['class_ids'][i])]
        score = float(pred['scores'][i])

        cv2.putText(img, '{}, {:.3f}'.format(obj, score),
                    (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 2)

    return img