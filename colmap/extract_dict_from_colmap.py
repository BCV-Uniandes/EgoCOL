import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
from os.path import exists
import pickle 
import argparse
import tqdm

def extract_dict(args):
    total_registered = 0
    total_images = 0
    total_poses = {}
    path_to_vq3d_clips = args.annotations_dir
    vq3d_clips = json.load(open(path_to_vq3d_clips, "r"))
    clips = vq3d_clips[args.split]
    for clip in clips:
        total_images += len(os.listdir(os.path.join(args.clips_dir,clip)))
        total_poses[clip] = {}
        if exists(os.path.join(args.input_dir_colmap, clip,"sparse/reg/images.txt")):
            with open(os.path.join(args.input_dir_colmap, clip,"sparse/reg/images.txt"), "r") as f:
                poses = f.readlines()[4:]
            for pose in tqdm.tqdm(poses):
                if "jpg" in pose:
                    params =  pose.split(" ")
                    qw, qx, qy, qz, tx, ty, tz = float(params[1]), float(params[2]), float(params[3]), float(params[4]), float(params[5]), float(params[6]), float(params[7])
                    r =  R.from_quat([qx, qy, qz, qw])
                    rot_matrix =  r.as_matrix()
                    extrinsics =  np.concatenate((rot_matrix, np.array([[tx],[ty],[tz]])), axis=1)
                    total_poses[clip][params[-1].strip("\n")] = extrinsics
                    total_registered+=1
            print(clip, len(total_poses[clip]))
        else:
            print("No registration for clip", clip)
    
    with open('colmap_ext_{}.pkl'.format(args.split), 'wb') as f:
        pickle.dump(total_poses, f)
    print("Registered Images/ Total Images: {}/{} ({} %) ".format(total_registered, total_images, (total_registered*100/total_images)))




if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        default='val'
        help="Set to extract. Train, val or test"
    )
    parser.add_argument(
        "--input_dir_colmap", 
        type=str, 
        default="/media/SSD5/ego4d/dataset/3d/v1/colmap"
        help="Path to the colmap output folder"
    )
    parser.add_argument(
        "--clips_dir",
        type=str,
        default='/media/SSD5/ego4d/dataset/3d/v1/clips_5fps_frames',
        help="Input folder with the clips.",
    )
    parser.add_argument(
        "--annotations_dir", 
        type=str, 
        default="/media/SSD5/ego4d/last_annotations/3d/all_clips_for_vq3d_v1.json"
        help="Path to the all_clips_for_vq3d_v1.json annotation file"
    )
    args = parser.parse_args()
    extract_dict(args)