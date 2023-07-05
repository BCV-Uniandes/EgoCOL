import os
import sys
import json
import argparse
import subprocess as sp
from pathlib import Path
import shutil


def gen_text(path, output_path):
    folders = sorted(os.listdir(path))
    with open(os.path.join(output_path,"register_frame_names_1fps.txt"), "w") as f:
        for folder_idx in range(0, len(folders), 5):
            f.write(folders[folder_idx]+"\n")

def gen_left_text(path, output_path):
    folders = sorted(os.listdir(path))
    with open(os.path.join(output_path,"register_frame_names_1fps.txt"), "r") as f:
        exists = f.readlines()
    exists = [i.strip("\n") for i in exists]
    with open(os.path.join(output_path,"register_frame_names_5fps.txt"), "w") as f:
        for folder_idx in range(len(folders)):
            if folders[folder_idx] not in exists:
                f.write(folders[folder_idx]+"\n")

def gen_intrinsics(ego_path, output_path):
    filename = os.path.join(ego_path, "fisheye_intrinsics.txt")
    with open(filename, "r") as f:
        lines = f.readlines()
    line = lines[0].split(" ") 
    intrinsics =[line[4], line[2], line[3], line[5], line[6]]
    intrinsics = ",".join(intrinsics)
    with open(os.path.join(output_path, 'fisheye_intrinsics.txt'), 'w') as f:
            f.write(intrinsics)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_poses_dir",
        type=str,
        default='/media/SSD5/ego4d/dataset/3d/v1/clips_camera_poses_5fps',
        help="Input folder with the clips camera poses 5fps",
    )
    parser.add_argument(
        "--clips_dir",
        type=str,
        default='/media/SSD5/ego4d/dataset/3d/v1/clips_5fps_frames',
        help="Input folder with the clips.",
    )

    parser.add_argument(
        "--path_vocab_tree",
        type=str,
        default="/media/SSD5/ego4d/dataset/3d/v1/colmap"
        help="path to the vocab_tree file for matching in colmap vocab_tree_flickr100K_words256K.bin",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default='/media/SSD6/ego4d/dataset/3d/colmap',
        help="Output folder.",
    )


    args = parser.parse_args()
    input_dir=args.input_poses_dir
    output_dir_root=args.output_dir
    clips_dir = args.clips_dir
    for clip_uid in os.listdir(input_dir):
        save_dir = output_dir_root
        save_dir_clip = os.path.join(save_dir, clip_uid)
        
        Path(save_dir_clip).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(save_dir_clip,"sparse")).mkdir(parents=True, exist_ok=True)
        dataset_dir =  os.path.join(clips_dir,clip_uid)
        scene_info_dir = save_dir_clip
        file_path = os.path.join(save_dir_clip,"sparse","reg","sparse.ply")
        gen_text(os.path.join(clips_dir,clip_uid), save_dir_clip)
        gen_left_text(os.path.join(clips_dir,clip_uid),save_dir_clip)
        gen_intrinsics(os.path.join(input_dir,clip_uid,"egovideo" ), save_dir_clip)
        vocab_path = os.path.join(args.path_vocab_tree, "vocab_tree_flickr100K_words256K.bin")
        cmd = ["./main.sh", f"{save_dir_clip}", f"{dataset_dir}", f"{scene_info_dir}", f"{file_path}", f"{vocab_path}"]

        # Get flag
        cmd.append("--features")
        cmd.append("--tree")
        cmd.append("--mapper")
        cmd.append("--register")
        o = sp.run(cmd)
        
