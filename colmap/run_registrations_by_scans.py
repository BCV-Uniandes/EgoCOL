import os
import sys
import json
import argparse
import subprocess as sp
from pathlib import Path
import shutil
from PIL import Image
from mpi4py import MPI
from tqdm import tqdm

def load_vq3d_annotation(filename):
    output = {}
    data = json.load(open(filename, 'r'))
    for video in data['videos']:
        scan_uid = video['scan_uid']
        for clip in video['clips']:
            output[clip['clip_uid']] = scan_uid
    return output

def gen_text(path, output_path):
    folders = sorted(os.listdir(path))
    clip = path.split("/")[-1]
    with open(os.path.join(output_path,"register_frame_names_1fps.txt"), "a") as f:
        for folder_idx in range(0, len(folders), 10):
            f.write(clip+"/"+folders[folder_idx]+"\n")

def gen_left_text(path, output_path):
    clip = path.split("/")[-1]
    folders = sorted(os.listdir(path))
    with open(os.path.join(output_path,"register_frame_names_1fps.txt"), "r") as f:
        exists = f.readlines()
    exists = [i.strip("\n") for i in exists]
    with open(os.path.join(output_path,"register_frame_names_5fps.txt"), "a") as f:
        for folder_idx in range(len(folders)):
            if (clip+"/"+folders[folder_idx]) not in exists:
                f.write(clip+"/"+folders[folder_idx]+"\n")

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
        "--output_dir",
        type=str,
        default='/media/SSD5/ego4d/dataset/3d/v1/colmap_scan',
        help="Output folder.",
    )
    parser.add_argument(
        "--camera_intrinsics_filename",
        type=str,
        default='/media/SSD5/ego4d/dataset/3d/v1/scan_to_intrinsics.json',
        help="Json file with all the camera intrinsics.",
    )
    parser.add_argument(
        "--query_filename",
        type=str,
        default='/media/SSD3/ego4d/dataset/3d/v1/annotations/vq3d_val.json',
        help="Input query annotation file vq3d_val.json. It depends on the split",
    )
    parser.add_argument(
        "--path_vocab_tree",
        type=str,
        default="/media/SSD5/ego4d/dataset/3d/v1/colmap"
        help="path to the vocab_tree file for matching in colmap vocab_tree_flickr100K_words256K.bin",
    )
    comm = MPI.COMM_WORLD
    size = comm.Get_size()  
    rank = comm.Get_rank()

    args = parser.parse_args()
    input_dir=args.input_poses_dir
    output_dir_root=args.output_dir
    save_dir = output_dir_root
    clips_dir = args.clips_dir
    all_intrinsics = json.load(open(args.camera_intrinsics_filename, 'r'))
    clip_uid_to_scan_uid = load_vq3d_annotation(args.query_filename)

    scan_name_to_uid = {
    'unict_Scooter mechanic_31': 'unict_3dscan_001',
    'unict_Baker_32': 'unict_3dscan_002',
    'unict_Carpenter_33': 'unict_3dscan_003',
    'unict_Bike mechanic_34': 'unict_3dscan_004',
    }
    for clip_uid in os.listdir(input_dir):
        scan_name = clip_uid_to_scan_uid[clip_uid]
        scan_uid = scan_name_to_uid[scan_name]
        egovideo_dir =  os.path.join(input_dir, clip_uid, "egovideo")
        im = Image.open(os.path.join(egovideo_dir,
                                     'color_distorted',
                                     os.listdir(os.path.join(egovideo_dir, 'color_distorted'))[0]))
        resolution = im.size 
        resolution_token = (str(resolution[0]),
                            str(resolution[1]))
        resolution_token_dir = str(resolution[0])+"_"+str(resolution[1])
        resolution_token = str(resolution_token)
        im.close()

        save_dir_scan = os.path.join(save_dir,scan_uid+"_"+resolution_token_dir)
        Path(save_dir_scan).mkdir(parents= True, exist_ok=True)
        Path(os.path.join(save_dir_scan, "sparse")).mkdir(parents=True, exist_ok= True)
        Path(os.path.join(save_dir_scan, "database")).mkdir(parents=True, exist_ok= True)
        if not os.path.exists(os.path.join(save_dir_scan, 'database', clip_uid)):
            os.symlink(os.path.join(os.path.join(clips_dir,clip_uid)),
                    os.path.join(save_dir_scan, 'database', clip_uid))
        gen_text(os.path.join(clips_dir,clip_uid), save_dir_scan)
        gen_left_text(os.path.join(clips_dir,clip_uid),save_dir_scan)
        gen_intrinsics(os.path.join(input_dir,clip_uid,"egovideo" ), save_dir_scan)
        
    vocab_path = os.path.join(args.path_vocab_tree, "vocab_tree_flickr100K_words256K.bin")
    scans = os.listdir(output_dir_root)
    length = len(scans)
    start = int(rank * int(length / size))
    end = int((start + (length / size) if (rank + 1) != size else length))
    iterable = range(start, end)
    if rank == 0:
        iterable = tqdm(range(start, end))
    for idx in iterable:
        scan = scans[idx]
        save_dir_scan = os.path.join(save_dir,scan)
        dataset_dir =  os.path.join(save_dir_scan,"database")
        file_path = os.path.join(save_dir_scan,"sparse","reg","sparse.ply")

        cmd = ["./main_by_scan.sh", f"{save_dir_scan}", f"{dataset_dir}", f"{save_dir_scan}", f"{file_path}", f"{vocab_path}"]

        # Get flag
        cmd.append("--features")
        cmd.append("--tree")
        cmd.append("--mapper")
        cmd.append("--register")
        o = sp.run(cmd)
        
