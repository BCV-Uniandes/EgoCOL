import os
import sys
import json
import argparse

import numpy as np

sys.path.append('API/')
from metrics import distL2
from metrics import angularError
from metrics import accuracy
from metrics import mAP
import matplotlib.pyplot as plt

sys.path.append('../annotation_API/API/')
from bounding_box import BoundingBox



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vq3d_results",
        type=str,
        default='data/vq3d_results/siam_rcnn_residual_kys_val.json',
        help="Camera pose folder"
    )
    args = parser.parse_args()


    # Visual Query 3D queries
    vq3d_queries = json.load(open(args.vq3d_results, 'r'))

    dl2 = distL2()
    dangle = angularError()
    acc = accuracy()
    mAP = mAP()

    all_l2 = []
    all_angles = []
    all_acc = []

    metrics = {'total':0,
               'l2':[],
               'angles':[],
               'success*': [],
               'success_overall': [],
               'total_wQframe_pose':0,
               'total_3d_estimation':0,
               'd*':[],
               'c*':[],
               'd':[],
               'c':[], 
               'mIoU':[]
              }

    cpt_valid_queries = 0
    for video in vq3d_queries['videos']:
        video_uid = video['video_uid']
        for clip in video['clips']:
            clip_uid = clip['clip_uid']
            for ai, annot in enumerate(clip['annotations']):
                if not annot: continue
                for qset_id, qset in annot['query_sets'].items():

                    metrics['total']+=1

                    if not 'pred_3d_vec_world' in qset: continue

                    pred_t = np.array(qset['pred_3d_vec_world'])
                    gt_t = np.array(qset['gt_3d_vec_world_1'])

                    # compute L2 metric with first annotation
                    l2error = dl2.compute(pred_t, gt_t)
                    metrics['l2'].append(l2error)

                    # compute accuracy with the two bounding boxes
                    box1 = BoundingBox(qset['3d_annotation_1'])

                    box2 = BoundingBox(qset['3d_annotation_2'])
                    # a (boolen) si d<6
                    # d: distancia entre los centroides
                    # c: límite para considerar que es exitoso
                    a, d, c = acc.compute(pred_t, box1, box2)
                    metrics['success*'].append(a)
                    metrics['d*'].append(d)
                    metrics['c*'].append(c)

                    IoU = mAP.compute_IoU(pred_t, box1, box2)
                    metrics["mIoU"].append(IoU)

                    # count total
                    metrics['total_3d_estimation']+=1

                    # count total and angular error with Query frame pose
                    if 'pred_3d_vec' in qset:
                        # compute angular metric with first annotation
                        pred_3d_vec = np.array(qset['pred_3d_vec'])
                        gt_3d_vec = np.array(qset['gt_3d_vec_1'])
                        angleerror = dangle.compute(pred_3d_vec, 
                                                    gt_3d_vec)
                        metrics['angles'].append(angleerror)

                        metrics['total_wQframe_pose']+=1
                        metrics['success_overall'].append(a)
                        metrics['d'].append(d)
                        metrics['c'].append(c)



    print('total number of queries: ', metrics['total'])
    print('queries with 3D estimation: ', metrics['total_3d_estimation'])
    print('queries with poses for both RT and QF: ', metrics['total_wQframe_pose'])
    print(' ')
    avg_l2 = np.mean(metrics['l2'])
    # =================== Gráficas ========================
    
    #bars = plt.bar(range(len(metrics['l2'])), metrics["l2"])
    #ax = plt.gca()
    #ax.get_xaxis().set_visible(False)
    #rects = ax.patches
    #for rect, label in zip(rects, metrics["l2"]):
    #    height = rect.get_height()
    #    ax.text(
    #        rect.get_x() + rect.get_width() / 2, height, round(label,1), ha="center", va="bottom"
    #    )
    #plt.title("l2 distances for query")
    #plt.xlabel("Query")
    #plt.ylabel("L2 Distance")
    #plt.savefig("/media/SSD0/mcescobar/episodic-memory/VQ3D/VQ3D/scripts/l2_r.png")
    #plt.close()
    
    plt.bar(range(len(metrics['d*'])), metrics["d*"], width=0.4, label="d")
    plt.plot(range(len(metrics['c*'])), metrics["c*"], label="c", marker="*", color="red")
    plt.title("Successful for query")
    plt.xlabel("Query")
    plt.ylabel("Distance")
    plt.savefig("/media/SSD0/mcescobar/episodic-memory/VQ3D/VQ3D/scripts/dc.png")
    # ======================= continua métricas
    avg_angle = np.mean(metrics['angles'])
    success_star = np.sum(metrics['success*']) / metrics['total_3d_estimation'] * 100.0
    success = np.sum(metrics['success_overall']) / metrics['total'] * 100.0
    avg_IoU = np.mean(metrics["mIoU"])
    print('L2: ', avg_l2)
    print('angular: ', avg_angle)
    print('Success* : ', success_star)
    print('Success : ', success)
    print('QwP ratio : ', metrics['total_wQframe_pose'] / metrics['total'] * 100.0)
    print('c*: disp', np.std(metrics["c*"]))
    print('mIoU:', avg_IoU)



