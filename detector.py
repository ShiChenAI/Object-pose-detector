import os
import cv2
from tqdm import tqdm
from module import ImageProcessor
from utils.torch_utils import select_device
from utils.utils import display_results

def main():
    mode = 0
    if mode == 0:
        #ip = ImageProcessor('./cfg/taisei_efficientdet.yml')
        #ip = ImageProcessor('./cfg/taisei_yolov4.yml')
        ip = ImageProcessor('./data/fukushima_yolov3.yml')
        #ip.process(video_src='0', show_results=True)
        device = select_device('0', batch_size=1)
        #img_show, scene_graph = ip.process(images=['/mnt/database/Dataset/SAVEWORK-TEST/2021_3/MAH02642_224.jpg'], device=device, show_results=True)
        #ip.process(images=['/mnt/database/Dataset/SAVEWORK-TEST/1111_3/MAH02669_234.jpg'], show_results=True)
        img_show, scene_graph = ip.process(images=['/mnt/database/Dataset/FUKUSHIMA_2/4040_7/MAH02615_37.jpg'], device=device, show_results=True)
        print(scene_graph)

if __name__ == '__main__':
    main()