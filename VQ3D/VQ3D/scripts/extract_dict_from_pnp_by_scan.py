import os
import sys
import json
import h5py
import torch
import argparse
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import pickle
sys.path.append('API/')
from get_query_3d_ground_truth import VisualQuery3DGroundTruth

def extract_dict(args):
    total = 0
    clip_names = []
    helper = VisualQuery3DGroundTruth()
    total_poses = {}
    path_to_vq3d_clips = "/media/SSD5/ego4d/last_annotations/3d/all_clips_for_vq3d_v1.json"
    vq3d_clips = json.load(open(path_to_vq3d_clips, "r"))
    clips = vq3d_clips[args.split]
    clips_valid = 0
    scans = os.listdir("/media/SSD5/ego4d/dataset/3d/v1/colmap_scan_test")
    for scan in scans:
        total_poses[scan] = {}
        clips = os.listdir(os.path.join("/media/SSD5/ego4d/dataset/3d/v1/colmap_scan_test", scan,"database"))
        for clip in clips:
            dirname = os.path.join(args.input_dir, clip, 'egovideo')
            if not os.path.isdir(dirname):
                print("NO PATH")
                continue
            poses = helper.load_pose(dirname)
            if poses is None: 
                print("No poses")
                continue
            T, valid_pose = poses
            total += np.sum(valid_pose)
            print(clip, np.sum(valid_pose))
            if np.sum(valid_pose)>0:
                clip_names.append(clip)
                clips_valid+=1
            for i, (T, valid_pose) in enumerate(zip(T, valid_pose)):
                if valid_pose:
                    total_poses[scan][os.path.join(clip,'color_%07d.jpg'%i)] = T[:3]
    with open('/media/SSD0/mcescobar/episodic-memory/colmap/pnp_ext_test_by_scan.pkl', 'wb') as f:
        pickle.dump(total_poses, f)
    print("Clips with valid poses {}/69".format(clips_valid))
    print("PnP images/Total images: {}/158023 ({})".format(total, (total*100/158023)))
    print(clip_names)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        default='test'
    )
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="/media/SSD5/ego4d/dataset/3d/v1/clips_camera_poses_5fps_test"
    )
    args = parser.parse_args()
    extract_dict(args)