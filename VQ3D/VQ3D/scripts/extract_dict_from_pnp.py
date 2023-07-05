import os
import sys
import json
import torch
import tensorflow as tf
import argparse
import open3d as o3d
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import pickle
sys.path.append('API/')
from get_query_3d_ground_truth import VisualQuery3DGroundTruth

def camera_center_from_extrinsics(p: tf.Tensor):
    """Computes camera center from extrinsics. p is the 3 x 4 extrinsics."""
    r = p[:3, :3]
    t = p[:, -1][..., tf.newaxis]
    return tf.squeeze(-tf.linalg.inv(r) @ t, axis=1)

def open3d_outlier_removal(points, nb_neighbors=5, std_ratio=4):
    point_clouds = []
    for pose in points.values():
        center = camera_center_from_extrinsics(pose).numpy()
        point_clouds.append(center)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_clouds)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    new_points = {}
    for i, tupl in enumerate(zip(points.keys(), points.values())):
        key, value = tupl[0], tupl[1]
        if i in ind:
            new_points[key]=value
    print("Original: {}, Filtered: {}".format(len(points), len(new_points)))
    return new_points

def extract_dict(args):
    total_clips = 0
    total_frames = 0
    total = 0
    clip_names = []
    helper = VisualQuery3DGroundTruth()
    total_poses = {}
    path_to_vq3d_clips = "/media/SSD5/ego4d/last_annotations/3d/all_clips_for_vq3d_v1.json"
    vq3d_clips = json.load(open(path_to_vq3d_clips, "r"))
    clips = vq3d_clips[args.split]
    clips_valid = 0
    for clip in clips:
        total_clips+=1
        total_poses[clip] = {}
        dirname = os.path.join(args.input_dir, clip, 'egovideo')
        if not os.path.isdir(dirname):
            print("NO PATH")
            continue
        breakpoint()
        poses = helper.load_pose(dirname)
        if poses is None: 
            print("No poses")
            continue
        T, valid_pose = poses
        
        print(clip, np.sum(valid_pose))
        #Descomentar para correr sin filtrar
        #total += np.sum(valid_pose)
        #if np.sum(valid_pose)>0:
        #    clip_names.append(clip)
        #    clips_valid+=1
        for i, (T, valid_pose) in enumerate(zip(T, valid_pose)):
            if valid_pose:
                total_poses[clip]['color_%07d.jpg'%i] = T[:3]
        
        # Filtrar
        new_points =  open3d_outlier_removal(total_poses[clip])
        if len(new_points)>0:
            clips_valid+=1
            total+=len(new_points)

    #with open('/media/SSD0/mcescobar/episodic-memory/colmap/pnp_ext_test.pkl', 'wb') as f:
    #    pickle.dump(total_poses, f)
    print("Clips with valid poses {}/69 ({})".format(clips_valid, (clips_valid*100/69)))
    print("PnP images/Total images: {}/158023 ({})".format(total, (total*100/158023)))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        default='val'
    )
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="/media/SSD5/ego4d/dataset/3d/v1/clips_camera_poses_5fps"
    )
    args = parser.parse_args()
    extract_dict(args)