import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description='results validation.')
    parser.add_argument('--file_dir', type=str, default='/mnt/database/Experiments/20210119_fukushima_2/-1/', help='results files.')
    parser.add_argument('--gt_dir', type=str, default='/mnt/database/Dataset/FUKUSHIMA_2/', help='ground truth files.')
    parser.add_argument('--output_dir', default='/mnt/database/Experiments/20210119_fukushima_2/', type=str, help='output dir')

    return parser.parse_args()

def calculate_results(data, target, distance='0'):
    if distance == '0':
        tp = data['{}_tp'.format(target)].sum()
        fp = data['{}_fp'.format(target)].sum()
        tn = data['{}_tn'.format(target)].sum()
        fn = data['{}_fn'.format(target)].sum()
        precision = 0 if tp == 0 else tp / (tp + fp)
        recall = 0 if tp == 0 else tp / (tp + fn)
    else:
        sub_data = data[data['distance']==distance]
        tp = sub_data['{}_tp'.format(target)].sum()
        fp = sub_data['{}_fp'.format(target)].sum()
        tn = sub_data['{}_tn'.format(target)].sum()
        fn = sub_data['{}_fn'.format(target)].sum()
        precision = 0 if tp == 0 else tp / (tp + fp)
        recall = 0 if tp == 0 else tp / (tp + fn)

    return precision, recall, tp, fp, tn, fn

def batch_process(file_dir, gt_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 
    checkpoint_folders = os.listdir(file_dir)
    best_precision_cp = ['', 0, 0]
    best_recall_cp = ['', 0, 0]
    for checkpoint_folder in tqdm(checkpoint_folders):
        checkpoint_path = os.path.join(file_dir, checkpoint_folder)
        result_file = os.path.join(checkpoint_path, 'results.txt')
        output_file = os.path.join(output_dir, '{}_performance.txt'.format(checkpoint_folder))
        precision, recall = process(result_file, gt_dir, output_file)
        if precision > best_precision_cp[1]:
            best_precision_cp[0] = checkpoint_folder
            best_precision_cp[1] = precision
            best_precision_cp[2] = recall
        if recall > best_recall_cp[2]:
            best_recall_cp[0] = checkpoint_folder
            best_recall_cp[1] = precision
            best_recall_cp[2] = recall

    print('Best precision:')
    print(best_precision_cp)
    print('Best recall:')
    print(best_recall_cp)

def process(file_dir, gt_dir, output_file):
    data = pd.read_csv(file_dir, 
                       names=['folder', 'helmet_u', 'helmet_nu', 'mask_u', 'mask_nu', 'fullface_u', 'fullface_nu'],
                       index_col='folder')
    #print(data)
    re_data = pd.DataFrame(columns=['folder', 'distance', 
                                    'helmet_tp', 'helmet_fp', 'helmet_tn', 'helmet_fn', 
                                    'mask_tp', 'mask_fp', 'mask_tn', 'mask_fn', 
                                    'fullface_tp', 'fullface_fp', 'fullface_tn', 'fullface_fn'],
                           index=['folder'])
    for idx, row in data.iterrows():
        gt_file = os.path.join(gt_dir, '{}.xml'.format(idx))
        with open(gt_file, 'r') as g:
            tree = ET.parse(g)
            root = tree.getroot()
            img_count = int(root.find('imgcount').text)
            spl_count = int(root.find('splcount').text)
            distance = root.find('distance').text
            helmet_tp = 0
            helmet_fp = 0
            helmet_tn = 0
            helmet_fn = 0
            mask_tp = 0
            mask_fp = 0
            mask_tn = 0
            mask_fn = 0
            fullface_tp = 0
            fullface_fp = 0
            fullface_tn = 0
            fullface_fn = 0
            for obj in root.iter('object'):
                name = obj.find('name').text
                gt_u = int(obj.find('use').text)
                gt_nu = int(obj.find('notuse').text)
                p_u = int(row['{}_u'.format(name)])
                p_nu = int(row['{}_nu'.format(name)])

                if name == 'helmet': 
                    helmet_fp = 0 if p_u - gt_u < 0 else p_u - gt_u
                    helmet_tp = min(p_u, gt_u)
                    helmet_fn = 0 if p_nu - gt_nu < 0 else p_nu - gt_nu
                    helmet_tn = min(p_nu, gt_nu)
                elif name == 'mask': 
                    mask_fp = 0 if p_u - gt_u < 0 else p_u - gt_u
                    mask_tp = min(p_u, gt_u)
                    mask_fn = 0 if p_nu - gt_nu < 0 else p_nu - gt_nu
                    mask_tn = min(p_nu, gt_nu)
                elif name == 'fullface': 
                    fullface_fp = 0 if p_u - gt_u < 0 else p_u - gt_u
                    fullface_tp = min(p_u, gt_u)
                    fullface_fn = 0 if p_nu - gt_nu < 0 else p_nu - gt_nu
                    fullface_tn = min(p_nu, gt_nu)
            re_data = re_data.append(pd.DataFrame({'folder': idx, 'distance': distance, 
                                                   'helmet_tp': helmet_tp, 'helmet_fp': helmet_fp, 'helmet_tn': helmet_tn, 'helmet_fn': helmet_fn,
                                                   'mask_tp': mask_tp, 'mask_fp': mask_fp, 'mask_tn': mask_tn, 'mask_fn': mask_fn,
                                                   'fullface_tp': fullface_tp, 'fullface_fp': fullface_fp, 'fullface_tn': fullface_tn, 'fullface_fn': fullface_fn},
                                     index=['folder']))

    re_data = re_data.dropna()
    #print(re_data)
    
    final_precision = 0
    final_recall = 0
    with open(output_file, 'w') as f:
        for dis in ['0', '3', '5', '7']:
            overall_tp = 0
            overall_fp = 0
            overall_tn = 0
            overall_fn = 0
            if dis == '0':
                f.write('overall: \n')
            else:
                f.write('{}m: \n'.format(dis))
            for target in ['helmet', 'mask', 'fullface']:
                precision, recall, tp, fp, tn, fn = calculate_results(re_data, target, dis)
                line = '{0}, {1}, {2}, {3}, {4}, {5}'.format(precision, recall, tp, fp, tn, fn)
                f.write(line)
                f.write('\n')
                overall_tp += tp
                overall_fp += fp
                overall_tn += tn
                overall_fn += fn
            overall_precision = 0 if overall_tp == 0 else overall_tp / (overall_tp + overall_fp)
            overall_recall = 0 if overall_tp == 0 else overall_tp / (overall_tp + overall_fn)
            line = '{0}, {1}, {2}, {3}, {4}, {5}'.format(overall_precision, overall_recall, overall_tp, overall_fp, overall_tn, overall_fn)
            f.write(line)
            f.write('\n')
            if dis == '0':
                final_precision = overall_precision
                final_recall = overall_recall

        f.flush()
    
    return final_precision, final_recall

def main():
    args = get_args()
    file_dir = args.file_dir
    gt_dir = args.gt_dir
    output_dir = args.output_dir
    batch_process(file_dir, gt_dir, output_dir)
    


if __name__ == '__main__':
    main()