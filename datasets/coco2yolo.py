import os
import shutil
import tqdm
import numpy as np
import yaml
import argparse
from pycocotools.coco import COCO
import itertools

def get_args():
    parser = argparse.ArgumentParser('Convert dataset from voc format to coco format.')

    # Common arguments
    parser.add_argument('--cfg', type=str, help='dataset configuration file path') 
    parser.add_argument('--coco-dir', type=str, help='image directory')
    parser.add_argument('--output-dir', type=str, help='output directory')

    args = parser.parse_args()
    print(args)

    return args

def coco_to_yolo_id(categories):
    coco_names = np.array(categories)
    yolo_names = np.array(categories)

    return yolo_names.tolist(), [list(coco_names[i] == yolo_names).index(True) if any(coco_names[i] == yolo_names) else None for i in range(coco_names.size)]

def convert(args, categories, dataset):
    images_dir = os.path.join(args.coco_dir, dataset)
    coco_ann_file = os.path.join(args.coco_dir, 'annotations/instances_{}.json'.format(dataset))
    coco = COCO(coco_ann_file)
    cat_names, yolo_ids = coco_to_yolo_id(categories)
    dataset_output_dir = os.path.join(args.output_dir, dataset)
    if not os.path.exists(dataset_output_dir):
        os.makedirs(dataset_output_dir)
    img_txt = os.path.join(args.output_dir, '{}.txt'.format(dataset))

    with open(img_txt, 'w') as f:
        cat_ids = coco.getCatIds(catNms=cat_names)
        for cat_id in cat_ids:
            yolo_id = yolo_ids[cat_id-1]
            img_ids = coco.getImgIds(catIds=cat_id)

            cat_anns_n = 0
            for img_id in img_ids:
                img = coco.loadImgs(ids=img_id)
                img_name = img[0]['file_name']
                if img_name == '1.jpg':
                    aaa = 0
                image_width = img[0]['width']
                image_height = img[0]['height']
                img_src = os.path.join(images_dir, img_name)
                img_dst = os.path.join(dataset_output_dir, img_name)
                shutil.copy(img_src, img_dst)
                f.write(img_dst)
                f.write(os.linesep)
                img_dst_txt = os.path.splitext(img_dst)[0] + '.txt'
                with open(img_dst_txt, 'a') as tf:
                    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_id)
                    anns = coco.loadAnns(ann_ids)
                    cat_anns_n += len(anns)
                    for ann in anns:
                        x_top_left = ann['bbox'][0]
                        y_top_left = ann['bbox'][1]
                        bbox_width = ann['bbox'][2]
                        bbox_height = ann['bbox'][3]

                        x_center = x_top_left + bbox_width / 2
                        y_center = y_top_left + bbox_height / 2

                        # darknet annotation format
                        #  <object-class> <x_center> <y_center> <width> <height>
                        a = x_center / image_width
                        b = y_center / image_height
                        c = bbox_width / image_width
                        d = bbox_height / image_height
                        
                        tf.write('{0} {1:.6} {2:.6} {3:.6} {4:.6}\n'.format(yolo_id, a, b, c, d))
                        #print(f"{yolo_id} {a:.6f} {b:.6f} {c:.6f} {d:.6f}", file=tf)

            print('Category: {0}={1}, anns: {2}, COMPLETED'.format(yolo_id, cat_names[yolo_id], cat_anns_n))

if __name__ == '__main__':
    args = get_args()
    
    categories = []
    with open(args.cfg) as f:
        categories = yaml.load(f, Loader=yaml.FullLoader)['names']
    print(categories)

    for dataset in ['train', 'val']:
        convert(args, categories, dataset)