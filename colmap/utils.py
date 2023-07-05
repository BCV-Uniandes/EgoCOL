import os
import json
from pydoc import cli
from readline import insert_text
from PIL import Image

def gen_text(path, output_path):
    folders = sorted(os.listdir(path))
    with open(os.path.join(output_path,"images.txt"), "w") as f:
        for folder in folders:
            f.write(folder+"\n")

def gen_intrinsics(ego_path, output_path):
    filename = os.path.join(ego_path, "fisheye_intrinsics.txt")
    breakpoint()
    with open(filename, "r") as f:
        lines = f.readlines()
    line = lines[0].split(" ") 
    intrinsics =[line[4], line[2], line[3], line[5], line[6]]
    intrinsics = ",".join(intrinsics)
    with open(os.path.join(output_path, 'fisheye_intrinsics.txt'), 'w') as f:
            f.write(intrinsics)