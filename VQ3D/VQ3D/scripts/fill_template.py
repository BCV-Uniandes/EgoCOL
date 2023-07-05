import os
import sys
import json
import h5py
import torch
import argparse
import numpy as np

from typing import Any, Dict, List, Optional, Tuple

sys.path.append('API/')
from get_query_3d_ground_truth import VisualQuery3DGroundTruth

def parse_VQ2D_queries(filename: str) -> Dict:
    output = {}
    data = json.load(open(filename, 'r'))
    for video in data['videos']:
        video_uid = video['video_uid']
        if video_uid not in output:
            output[video_uid]={}
        for clip in video['clips']:
            clip_uid = clip['clip_uid']
            if clip_uid not in output[video_uid]:
                output[video_uid][clip_uid]={}
            for ai, annot in enumerate(clip['annotations']):
                if ai not in output[video_uid][clip_uid]:
                    output[video_uid][clip_uid][ai]={}
                for qset_id, qset in annot['query_sets'].items():
                    if not qset["is_valid"]:
                        continue
                    output[video_uid][clip_uid][ai][qset_id] = {
                        "query_frame": qset["query_frame"],
                        "object_title": qset["object_title"],
                        "visual_crop": qset["visual_crop"],
                        "response_track": qset["response_track"],
                    }
    return output


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_filename",
        type=str,
        default='data/vq3d_results/siam_rcnn_residual_kys_val.json',
        help="Camera pose folder"
    )
    parser.add_argument(
        "--vq3d_queries",
        type=str,
        default='data/vq3d_val_wgt.json',
        help="VQ3D query file"
    )
    parser.add_argument(
        "--vq2d_queries",
        type=str,
        default='data/vq_val.json',
        help="VQ3D query file"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default='data/clips_from_videos_camera_poses/',
        help="Camera pose folder"
    )
    
    args = parser.parse_args()
    root_dir = args.input_dir

    output_filename = args.output_filename

    # Visual Query 3D queries
    vq3d_template = json.load(open(args.vq3d_queries, 'r'))

    # Visual Query 2D results
    vq2d_queries = json.load(open(args.vq2d_queries, 'r'))
    
    query_matching_filename=f'data/mapping_vq2d_to_vq3d_queries_annotations_test.json'
    query_matching = json.load(open(query_matching_filename, 'r'))

    helper = VisualQuery3DGroundTruth()

    cpt_valid_queries = 0
    for video_index ,video in enumerate(vq3d_template['videos']):
        video_uid = video['video_uid']
        for clip_id, clip in enumerate(video['clips']):
            clip_uid = clip['clip_uid']
            for ai, annot in enumerate(clip['annotations']):
                if not annot: 
                    print("VACIO")
                    continue
                for qset_id, qset in annot['query_sets'].items():
                    mapping_ai=query_matching[video_uid][clip_uid][str(ai)][qset_id]['ai']
                    mapping_qset_id=query_matching[video_uid][clip_uid][str(ai)][qset_id]['qset_id']
                    query_pred=vq2d_queries[video_uid][clip_uid][str(mapping_ai)][mapping_qset_id]
                    query_frame = query_pred["query_frame"]            
                
                    # get poses
                    dirname = os.path.join(root_dir, clip_uid, 'egovideo')

                    if not os.path.isdir(dirname): 
                        print("No egovideo folder")
                        continue

                    poses = helper.load_pose(dirname)
                    if poses is None: 
                        print("No poses found")
                        continue
                    T, valid_pose = poses
                    if np.sum(valid_pose)>20:
                        print(clip_uid)
                        breakpoint()
                        np.where(valid_pose)

                    if "pred_3d_vec_world" in query_pred:
                        pred_3d_vec_world = query_pred["pred_3d_vec_world"]
                    else:
                        pred_3d_vec_world = None
                    if "pred_3d_vec" in query_pred:
                        pred_3d_vec = query_pred["pred_3d_vec"]
                    else:
                        pred_3d_vec =  None
                    
                    if valid_pose[query_frame]:
                        query_frame_pose = T[query_frame].tolist()
                    else:
                        query_frame_pose = None
                    vq3d_template["videos"][video_index]["clips"][clip_id]["annotations"][ai]["query_sets"][qset_id]["pred_3d_vec_world"] = pred_3d_vec_world
                    vq3d_template["videos"][video_index]["clips"][clip_id]["annotations"][ai]["query_sets"][qset_id]["pred_3d_vec_query_frame"] = pred_3d_vec
                    vq3d_template["videos"][video_index]["clips"][clip_id]["annotations"][ai]["query_sets"][qset_id]["query_frame_pose"] = query_frame_pose
                    
#json.dump(vq3d_template, open(output_filename, 'w'))