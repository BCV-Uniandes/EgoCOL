import numpy as np
import json

real_camera_poses = json.load(open("/media/SSD0/mcescobar/episodic-memory/VQ3D/data/all_clips_camera_poses.json", "r"))["f91c7008-f21c-4047-bfc1-d937787665e5"]["camera_poses"]
propio_camera_poses = np.load("/media/SSD5/ego4d/dataset/3d/v1/clips_camera_poses_5fps/f91c7008-f21c-4047-bfc1-d937787665e5/egovideo/poses_reloc/camera_poses_pnp.npy")

real_camera_poses = np.array(real_camera_poses)

comparacion =real_camera_poses==propio_camera_poses
for i in range(len(comparacion)):
    if not comparacion[i].all():
        print("===================",i, "================")
        print(real_camera_poses[i], propio_camera_poses[i])

for i in range(len(real_camera_poses)):
    if (real_camera_poses[50]==propio_camera_poses[i]).all():
        breakpoint()
