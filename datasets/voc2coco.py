import sys
import os
import argparse
import json
import yaml
import xml.etree.ElementTree as ET
import shutil
from PIL import Image
import random

def get_args():
    parser = argparse.ArgumentParser('Convert dataset from voc format to coco format.')

    # Common arguments
    parser.add_argument('--start-id', type=int, default=1, help='start bounding box ID')
    parser.add_argument('--cfg', type=str, help='dataset configuration file path') 
    parser.add_argument('--images-dir', type=str, help='image directory') 
    parser.add_argument('--annotations-dir', type=str, help='annotation directory') 
    parser.add_argument('--dataset-list', type=str, help='dataset list file')
    parser.add_argument('--output-dir', type=str, help='output directory') 
    parser.add_argument('--split-ratio', type=int, default=95, help='split ration of train/val dataset') 

    args = parser.parse_args()
    print(args)

    return args

def get(root, name):
    vars = root.findall(name)
    return vars

def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars

def get_filename_as_int(filename):
    try:
        filename = os.path.splitext(filename)[0]
        return int(filename)
    except:
        raise NotImplementedError('Filename %s is supposed to be an integer.'%(filename))

def is_jpg(filename):
    try:
        i=Image.open(filename)
        return i.format =='JPEG'
    except IOError:
        return False

def random_index(rate):
    start = 0
    index = 0
    randnum = random.randint(1, sum(rate))

    for index, scope in enumerate(rate):
        start += scope
        if randnum <= start:
            break

    return index

def get_dataset(split_ratio):
    arr = ['train', 'val']
    rate = [split_ratio, 100-split_ratio]

    return arr[random_index(rate)]


def convert(args, categories):
    xml_dirs = []
    if args.dataset_list is None:
        xml_dirs.append(args.annotations_dir)
    else:
        with open(args.dataset_list, 'r') as f:
            for line in f:
                xml_dirs.append(os.path.join(args.annotations_dir, line.strip()))

    dest_ann_path = os.path.join(args.output_dir, 'annotations')
    if not os.path.exists(dest_ann_path):
        os.makedirs(dest_ann_path)
    train_json_file = os.path.join(dest_ann_path, 'instances_train.json')
    val_json_file = os.path.join(dest_ann_path, 'instances_val.json')
    train_json_dict = {'images':[], 'type': 'instances', 'annotations': [], 'categories': []}
    val_json_dict = {'images':[], 'type': 'instances', 'annotations': [], 'categories': []}
    image_id = args.start_id
    bnd_id = args.start_id

    for xml_dir in xml_dirs:
        for _, _, xml_filenames in os.walk(xml_dir):
            for xml_filename in xml_filenames:
                dataset = get_dataset(args.split_ratio)
                if dataset == 'train':
                    dest_img_dir = os.path.join(args.output_dir, 'train')
                elif dataset == 'val':
                    dest_img_dir = os.path.join(args.output_dir, 'val')
                
                if not os.path.exists(dest_img_dir):
                    os.makedirs(dest_img_dir)
                
                xml_path = os.path.join(xml_dir, xml_filename)
                tree = ET.parse(xml_path)
                root = tree.getroot()

                src_img_path = root.find('path').text
                dest_img_path = os.path.join(dest_img_dir, '{}.jpg'.format(image_id))
                shutil.copy(src_img_path, dest_img_path)
                if not is_jpg(dest_img_path):
                    os.remove(dest_img_path)
                    continue

                has_cat = False
                for obj in get(root, 'object'):
                    category = get_and_check(obj, 'name', 1).text
                    if category not in categories:
                        continue

                    has_cat = True
                    category_id = categories[category]
                    bndbox = get_and_check(obj, 'bndbox', 1)
                    xmin = int(get_and_check(bndbox, 'xmin', 1).text) - 1
                    ymin = int(get_and_check(bndbox, 'ymin', 1).text) - 1
                    xmax = int(get_and_check(bndbox, 'xmax', 1).text)
                    ymax = int(get_and_check(bndbox, 'ymax', 1).text)
                    assert(xmax > xmin)
                    assert(ymax > ymin)
                    o_width = abs(xmax - xmin)
                    o_height = abs(ymax - ymin)
                    ann = {'area': o_width*o_height, 
                        'iscrowd': 0, 
                        'image_id': image_id, 
                        'bbox':[xmin, ymin, o_width, o_height],
                        'category_id': category_id, 
                        'id': bnd_id, 
                        'ignore': 0,
                        'segmentation': []}

                    if dataset == 'train':
                        train_json_dict['annotations'].append(ann)
                    elif dataset == 'val':
                        val_json_dict['annotations'].append(ann)
                    
                    bnd_id += 1

                if has_cat:
                    size = get_and_check(root, 'size', 1)
                    width = int(get_and_check(size, 'width', 1).text)
                    height = int(get_and_check(size, 'height', 1).text)
                    image = {'file_name': '{}.jpg'.format(image_id), 
                            'height': height, 
                            'width': width,
                            'id':image_id}

                    if dataset == 'train':
                        train_json_dict['images'].append(image)
                    elif dataset == 'val':
                        val_json_dict['images'].append(image)

                    image_id += 1
                    print('Processed: {}'.format(image_id))

    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 
               'id': cid, 
               'name': cate}
        train_json_dict['categories'].append(cat)
        val_json_dict['categories'].append(cat)

    with open(train_json_file, 'w') as train_json_fp:
        train_json_str = json.dumps(train_json_dict)
        train_json_fp.write(train_json_str)

    with open(val_json_file, 'w') as val_json_fp:
        val_json_str = json.dumps(val_json_dict)
        val_json_fp.write(val_json_str)


if __name__ == '__main__':
    args = get_args()
    
    categories = {}
    with open(args.cfg) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader) 
        for i, category in enumerate(data_dict['names']):
            categories[category] = i + 1
    print(categories)

    convert(args, categories)