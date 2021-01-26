import json
import cv2
import numpy as np
import os
import yaml
import argparse
import json
import numpy as np
import glob
from PIL import Image
from PIL import ImageDraw
import io
import base64
import shutil
from sklearn.model_selection import train_test_split

def get_args():
    parser = argparse.ArgumentParser('Convert dataset from voc format to coco format.')

    # Common arguments
    parser.add_argument('--cfg', type=str, help='dataset configuration file path') 
    parser.add_argument('--input-dir', type=str, help='input directory')
    parser.add_argument('--output-dir', type=str, help='output directory')
    parser.add_argument('--split-ratio', type=float, default=0.2, help='split ration of train/val dataset') 

    args = parser.parse_args()
    print(args)

    return args

def png2jpg(png_path, jpg_path):
    img = cv2.imread(png_path, 0)
    w, h = img.shape[::-1]
    infile = png_path
    #outfile = os.path.splitext(infile)[0] + ".jpg"
    outfile = jpg_path
    img = Image.open(infile)
    #img = img.resize((int(w / 2), int(h / 2)), Image.ANTIALIAS)
    try:
        if len(img.split()) == 4:
            # prevent IOError: cannot write mode RGBA as BMP
            r, g, b, a = img.split()
            img = Image.merge("RGB", (r, g, b))
            img.convert('RGB').save(outfile, quality=100)
        else:
            img.convert('RGB').save(outfile, quality=100)
        return outfile
    except Exception as e:
        print("PNG转换JPG 错误", e)

def is_jpg(filename):
    try:
        i=Image.open(filename)
        return i.format =='JPEG'
    except IOError:
        return False

def img_data_to_pil(img_data):
    f = io.BytesIO()
    f.write(img_data)
    img_pil = Image.open(f)
    return img_pil

def img_data_to_arr(img_data):
    img_pil = img_data_to_pil(img_data)
    img_arr = np.array(img_pil)
    return img_arr

def img_b64_to_arr(img_b64):
    img_data = base64.b64decode(img_b64)
    img_arr = img_data_to_arr(img_data)
    return img_arr

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


class labelme2coco(object):
    def __init__(self, categories, labelme_json, save_json_path, save_img_path):
        '''
        :param labelme_json: 所有labelme的json文件路径组成的列表
        :param save_json_path: json保存位置
        '''
        self.labelme_json = labelme_json
        self.save_json_path = save_json_path
        self.save_img_path = save_img_path
        self.images = []
        self.categories = []
        self.annotations = []
        # self.data_coco = {}
        self.label = []
        for cat in categories:
            self.categories.append(self.categorie(cat))
            self.label.append(cat)
        self.annID = 1
        self.height = 0
        self.width = 0

        self.save_json()

    def data_transfer(self):
        for num, json_file in enumerate(self.labelme_json):
            with open(json_file, 'r') as fp:
                data = json.load(fp)  # 加载json文件
                self.images.append(self.image(json_file, data, num))
                for shapes in data['shapes']:
                    label = shapes['label']
                    """
                    if label not in self.label:
                        self.categories.append(self.categorie(label))
                        self.label.append(label)
                    """
                    if label == 'arrow_2':
                        label = 'arrow'
                    if label not in self.label:
                        continue
                    points = shapes['points']  # 这里的point是用rectangle标注得到的，只有两个点，需要转成四个点
                    self.annotations.append(self.annotation(points, label, num))
                    self.annID += 1

    def image(self, json_file, data, num):
        image = {}
        img = img_b64_to_arr(data['imageData'])  # 解析原图片数据
        # img=io.imread(data['imagePath']) # 通过图片路径打开图片
        # img = cv2.imread(data['imagePath'], 0)
        height, width = img.shape[:2]
        img = None
        image['height'] = height
        image['width'] = width
        image['id'] = num + 1
        # image['file_name'] = data['imagePath'].split('/')[-1]
        img_file = os.path.join(os.path.dirname(json_file), data['imagePath'])
        if not is_jpg(img_file):
            png2jpg(img_file, os.path.join(self.save_img_path, '{}.jpg'.format(os.path.splitext(os.path.basename(json_file))[0])))
        else:
            dest_img_path = os.path.join(self.save_img_path, os.path.join(self.save_img_path, '{}.jpg'.format(os.path.splitext(os.path.basename(json_file))[0])))
            shutil.copy(img_file, dest_img_path)
        #image['file_name'] = data['imagePath']
        image['file_name'] = '{}.jpg'.format(os.path.splitext(os.path.basename(json_file))[0])
        self.height = height
        self.width = width

        return image

    def categorie(self, label):
        categorie = {}
        categorie['supercategory'] = 'None'
        categorie['id'] = len(self.label) + 1  # 0 默认为背景
        categorie['name'] = label
        return categorie

    def annotation(self, points, label, num):
        annotation = {}
        annotation['segmentation'] = [list(np.asarray(points).flatten())]
        annotation['iscrowd'] = 0
        annotation['image_id'] = num + 1
        # annotation['bbox'] = str(self.getbbox(points)) # 使用list保存json文件时报错（不知道为什么）
        # list(map(int,a[1:-1].split(','))) a=annotation['bbox'] 使用该方式转成list
        annotation['bbox'] = list(map(float, self.getbbox(points)))
        annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3]
        # annotation['category_id'] = self.getcatid(label)
        annotation['category_id'] = self.getcatid(label)  # 注意，源代码默认为1
        annotation['id'] = self.annID
        return annotation

    def getcatid(self, label):
        for categorie in self.categories:
            if label == categorie['name']:
                return categorie['id']
        return 1

    def getbbox(self, points):
        # img = np.zeros([self.height,self.width],np.uint8)
        # cv2.polylines(img, [np.asarray(points)], True, 1, lineType=cv2.LINE_AA)  # 画边界线
        # cv2.fillPoly(img, [np.asarray(points)], 1)  # 画多边形 内部像素值为1
        polygons = points

        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):
        '''从mask反算出其边框
        mask：[h,w]  0、1组成的图片
        1对应对象，只需计算1对应的行列号（左上角行列号，右下角行列号，就可以算出其边框）
        '''
        # np.where(mask==1)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]

        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        return [left_top_c, left_top_r, right_bottom_c - left_top_c,
                right_bottom_r - left_top_r]  # [x1,y1,w,h]

    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def data2coco(self):
        data_coco = {}
        data_coco['images'] = self.images
        data_coco['categories'] = self.categories
        data_coco['annotations'] = self.annotations
        return data_coco

    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        # 保存json文件
        json.dump(self.data_coco, open(self.save_json_path, 'w'), indent=4, cls=MyEncoder)  # indent=4 更加美观显示

if __name__ == '__main__':
    args = get_args()
    
    categories = []
    with open(args.cfg) as f:
        categories = yaml.load(f, Loader=yaml.FullLoader)['names']
    print(categories)
    dest_ann_path = os.path.join(args.output_dir, 'annotations')
    if not os.path.exists(dest_ann_path):
        os.makedirs(dest_ann_path)

    labelme_json = glob.glob(os.path.join(args.input_dir, '*.json'))
    train_files, val_files = train_test_split(labelme_json, test_size=args.split_ratio, random_state=55)
    dest_train_img_path = os.path.join(args.output_dir, 'train')
    if not os.path.exists(dest_train_img_path):
        os.makedirs(dest_train_img_path)
    labelme2coco(categories, train_files, os.path.join(dest_ann_path, 'instances_train.json'), dest_train_img_path)
    dest_val_img_path = os.path.join(args.output_dir, 'val')
    if not os.path.exists(dest_val_img_path):
        os.makedirs(dest_val_img_path)
    labelme2coco(categories, val_files, os.path.join(dest_ann_path, 'instances_val.json'), dest_val_img_path)
    