"""
Simple test script for semantic segmentation
"""
import argparse
import os
from data_utils.S3DISDataLoader import ScannetDatasetWholeScene
from data_utils.indoor3d_util import g_label2color
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
# import provider
import numpy as np
from torch.serialization import add_safe_globals

# Add safe globals for numpy
add_safe_globals(['numpy.core.multiarray.scalar'])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

NUM_CLASSES = 13
BATCH_SIZE = 32
NUM_POINT = 4096
VOTE_NUMS=1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


def load_model():
    experiment_dir = 'log/sem_seg/pointnet2_sem_seg/'
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).to(device)
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', weights_only=False, map_location=device)  # Load with weights_only=False
    classifier.load_state_dict(checkpoint['model_state_dict'])  # Removed map_location parameter
    classifier = classifier.eval()
    return classifier

def inference(classifier,input_path,visual_dir):
    # breakpoint()
    test_area_index = os.path.basename(input_path).split('_')[1]
    TEST_DATASET_WHOLE_SCENE = ScannetDatasetWholeScene(os.path.dirname(input_path), split='test', test_area=int(test_area_index), block_points=NUM_POINT)

    with torch.no_grad():
        scene_id = TEST_DATASET_WHOLE_SCENE.file_list  #npy文件的文件名
        """截取Area_5_conferenceroom.npy前边的截取Area_5_conferenceroom"""
        scene_id = [x[:-4] for x in scene_id]
        """返回的是scene_points_list列表长度，列表中的元素是[4096,6]"""
        num_batches = len(TEST_DATASET_WHOLE_SCENE)

        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

        for batch_idx in range(num_batches):
            print("Inference [%d/%d] %s ..." % (batch_idx + 1, num_batches, scene_id[batch_idx]))
            total_seen_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_correct_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class_tmp = [0 for _ in range(NUM_CLASSES)]

            fout = open(os.path.join(visual_dir, scene_id[batch_idx] + '_pred.txt'), 'w')
            fout_gt = open(os.path.join(visual_dir, scene_id[batch_idx] + '_gt.txt'), 'w')

            whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx]      #[4096,6]
            whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[batch_idx]  #[4096,1]
            vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))  #[4096，13]全零矩阵
            for _ in tqdm(range(VOTE_NUMS), total=VOTE_NUMS):
                scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx]
                num_blocks = scene_data.shape[0]
                s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
                batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))

                batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT))

                for sbatch in range(s_batch_num):
                    start_idx = sbatch * BATCH_SIZE
                    end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                    real_batch_size = end_idx - start_idx
                    batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                    batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                    batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                    batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]
                    batch_data[:, :, 3:6] /= 1.0

                    #breakpoint()
                    torch_data = torch.Tensor(batch_data)
                    torch_data = torch_data.float().to(device)
                    torch_data = torch_data.transpose(2, 1)
                    seg_pred, _ = classifier(torch_data)
                    batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()
                    """ 进行vote """
                    vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                               batch_pred_label[0:real_batch_size, ...],
                                               batch_smpw[0:real_batch_size, ...])
            """ #获取最终的标签 """
            pred_label = np.argmax(vote_label_pool, 1)
            """计算指标"""
            for l in range(NUM_CLASSES):
                total_seen_class_tmp[l] += np.sum((whole_scene_label == l))
                total_correct_class_tmp[l] += np.sum((pred_label == l) & (whole_scene_label == l))
                total_iou_deno_class_tmp[l] += np.sum(((pred_label == l) | (whole_scene_label == l)))
                total_seen_class[l] += total_seen_class_tmp[l]
                total_correct_class[l] += total_correct_class_tmp[l]
                total_iou_deno_class[l] += total_iou_deno_class_tmp[l]

            iou_map = np.array(total_correct_class_tmp) / (np.array(total_iou_deno_class_tmp, dtype=np.float64) + 1e-6)
            print(iou_map)
            arr = np.array(total_seen_class_tmp)
            tmp_iou = np.mean(iou_map[arr != 0])
            print('----------------------------')
            """创建对应点云场景的.txt文件"""
            filename = os.path.join(visual_dir, scene_id[batch_idx] + '.txt')
            with open(filename, 'w') as pl_save:
                for i in pred_label:  #遍历每一行标签值
                    pl_save.write(str(int(i)) + '\n') #转换为字符串写入文件每行一个数字
                pl_save.close()
            for i in range(whole_scene_label.shape[0]): #遍历所有点如果有4096个点则遍历4096次每次则写入一行数据
                color = g_label2color[pred_label[i]]            #预测值颜色
                color_gt = g_label2color[whole_scene_label[i]]  #真实值颜色

                """ 每次写入一行数据，分别为 x y z r g b """
                fout.write('%f %f %f %d %d %d\n' % (
                    whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color[0], color[1],
                    color[2]))
                """ground truth也有一个可视化文件同样的操作"""
                fout_gt.write(
                    '%f %f %f %d %d %d\n' % (
                        whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color_gt[0],
                        color_gt[1], color_gt[2]))

            fout.close()
            fout_gt.close()

        IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6)
        iou_per_class_str = '------- IoU --------\n'
        for l in range(NUM_CLASSES):
            iou_per_class_str += 'class %s, IoU: %.3f \n' % (
                seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])),
                total_correct_class[l] / float(total_iou_deno_class[l]))

        print("Done!")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run point cloud semantic segmentation inference')
    parser.add_argument('--input_path', type=str, default='./data/Area_5_conferenceRoom_1.npy', help='Path to input point cloud file')
    parser.add_argument('--output_path', type=str, default='./results', help='Path to save output results')
    args = parser.parse_args()
    
    classfier=load_model()
    inference(classfier,args.input_path, args.output_path)