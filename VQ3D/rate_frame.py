import json
import os
import numpy as np
root = "/media/SSD5/ego4d/dataset/3d/v1/clips_camera_poses_5fps"
total = 0
good = 0
i=0
for clip in os.listdir(root):
    print(i+1)
    i+=1
    path = os.path.join(root, clip, "egovideo", "superglue_track","poses", "good_pose_reprojection.npy")
    poses = np.load(path)
    total += len(poses)
    good += np.sum(poses)

print("The rate is {}, Total: {}, Good: {}".format(good/total, total, good))

