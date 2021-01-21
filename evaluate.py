import os
import argparse
from tqdm import tqdm
import cv2
from module import ImageProcessor
from utils.utils import get_dir_name, split_path
from utils.utils import get_models_info, update_model_path, display_results
from utils.torch_utils import select_device

def get_args():
    parser = argparse.ArgumentParser(description='evaluation.')
    parser.add_argument('--cfg', type=str, default='./cfg/taisei_yolov4.yml', help='Configure file.')
    parser.add_argument('--validate_dir', type=str, default='/mnt/database/Experiments/20201208/testlist/', help='Image files.')
    parser.add_argument('--output_path', type=str, default='/mnt/database/Experiments/20210110_taisei_csp', help='Output path.')
    parser.add_argument('--models_dir', type=str, help='Model files.')
    parser.add_argument('--save_imgs', action='store_true', help='Save result images.')
    parser.add_argument('--fukushima', action='store_true', help='Processing Fukushima data.')
    #parser.add_argument('--result_file', type=str, default='/mnt/database/Experiments/20201208/d0/results.txt', help='Output path.')

    return parser.parse_args()

def evaluate_case(ip, validate_dir, output_path, coef, checkpoint, object_model, regressBoxes, clipBoxes, input_size, pose_model, device, save_imgs, is_fukushima):
    params = ip.get_params()
    result_dir = os.path.join(output_path, coef, checkpoint)
    result_file = os.path.join(result_dir, 'results.txt')

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    else:
        return
    for root_dir, _, file_names in os.walk(validate_dir):
        for file_name in file_names:
            validate_file = os.path.join(root_dir, file_name)
            with open(validate_file, 'r') as vf:
                dir_name = get_dir_name(validate_file)
                img_list = vf.readlines()
                if os.path.exists(result_file):
                    file_mode = 'a'
                else:
                    file_mode = 'w' 
                with open(result_file, mode=file_mode) as rf:
                    if is_fukushima:
                        using_count = {'helmet': 0, 'mask': 0, 'fullface': 0}
                        npu_count = {'helmet': 0, 'mask': 0, 'fullface': 0}
                    else:
                        using_count = {'helmet': 0, 'visor': 0, 'mask': 0, 'safetybelt': 0}
                        npu_count = {'helmet': 0, 'visor': 0, 'mask': 0, 'safetybelt': 0}
                    for img_name in tqdm(img_list):
                        save_dir = os.path.join(output_path, coef, checkpoint, 'output')
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        img_name = img_name.strip('\n')
                        file_dir, file_name = os.path.split(img_name)
                        dir1, dir2, _ = split_path(img_name)
                        save_dir = os.path.join(save_dir, dir2)
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)

                        frame = cv2.imread(img_name, cv2.IMREAD_COLOR)
                        
                        previous_poses = []
                        out, current_poses, ori_imgs, scene_graph = ip.process_frame(frame=frame, 
                                                                            input_size=input_size,
                                                                            object_model=object_model,
                                                                            pose_model=pose_model,
                                                                            previous_poses=previous_poses,
                                                                            device=device)
                        save_path = os.path.join(save_dir, 'output_{}'.format(file_name))
                        if save_imgs:
                            img_show = display_results(pred=out, 
                                       img=ori_imgs[0], 
                                       obj_list=params.obj_list, 
                                       colors=ip.obj_colors, 
                                       current_poses=current_poses,
                                       track=params.track)
                            cv2.imwrite(save_path, img_show)

                        for _, v in scene_graph.items():
                            for target in using_count.keys():
                                if target in v['vertices']:
                                    using_count[target] += 1
                                else:
                                    npu_count[target] += 1

                    if is_fukushima:
                        line = '{0},{1},{2},{3},{4},{5},{6}'.format(dir_name, 
                                                                    using_count['helmet'], npu_count['helmet'], 
                                                                    using_count['mask'], npu_count['mask'],
                                                                    using_count['fullface'], npu_count['fullface'])
                    else:
                        line = '{0},{1},{2},{3},{4},{5},{6},{7},{8}'.format(dir_name, 
                                                                            using_count['helmet'], npu_count['helmet'], 
                                                                            using_count['visor'], npu_count['visor'],
                                                                            using_count['mask'], npu_count['mask'],
                                                                            using_count['safetybelt'], npu_count['safetybelt'])
                    
                    rf.write(line)
                    rf.write('\n')
                    rf.flush()
def main():
    args = get_args()
    device = select_device('0', batch_size=1)
    ip = ImageProcessor(args.cfg)
    params = ip.get_params()   
    object_model, input_size, pose_model = ip.init_models(device)
    evaluate_case(ip, args.validate_dir, args.output_path, '-1', '-1', object_model, None, None, input_size, pose_model, device, args.save_imgs, args.fukushima)

if __name__ == '__main__':
    main()