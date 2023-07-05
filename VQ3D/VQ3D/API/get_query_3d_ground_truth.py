import os
import sys
import cv2
import json
import fnmatch
import numpy as np
import open3d as o3d
import tensorflow as tf
from typing import Any, Dict, List, Optional, Tuple

sys.path.append('../annotation_API/API/')
from bounding_box import BoundingBox

def camera_center_from_extrinsics(p: tf.Tensor):
    """Computes camera center from extrinsics. p is the 3 x 4 extrinsics."""
    r = p[:3, :3]
    t = p[:, -1][..., tf.newaxis]
    return tf.squeeze(-tf.linalg.inv(r) @ t, axis=1)

def open3d_outlier_removal(C_T_G, valid_pose, nb_neighbors=5, std_ratio=4):
    point_clouds = []
    mapping={}
    indx = 0
    for i, pose in enumerate(C_T_G):
        if np.allclose(pose, np.zeros((3, 4))):
            continue
        mapping[i] = indx
        center = camera_center_from_extrinsics(pose).numpy()
        point_clouds.append(center)
        indx+=1
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_clouds)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    valid_pose_vieja = valid_pose.copy()
    for i in range(len(valid_pose)):
        if valid_pose[i] and mapping[i] not in ind:
            valid_pose[i]=False
    print("Original: {}, Filtered: {}".format(sum(valid_pose_vieja), sum(valid_pose)))
    return valid_pose

class VisualQuery3DGroundTruth():
    def __init__(self):
        pass

    def load_pose(self, dirname: str, check_colmap=False):
        pose_dir = os.path.join(dirname, 'superglue_track', 'poses')
        if check_colmap:
            # Read just the clip branch
            # if os.path.isfile(os.path.join(pose_dir, "valid_poses_colmap_filtered.npy")) and os.path.isfile(os.path.join(pose_dir, "poses_colmap_filtered.npy")):
            #     print("Extracting from colmap computed from poses by clip branch")
            #     valid_pose_clip = np.load(os.path.join(pose_dir, "valid_poses_colmap_filtered.npy"))
            #     Ci_T_G_clip = np.load(os.path.join(pose_dir, "poses_colmap_filtered.npy"))
            #     return Ci_T_G_clip, valid_pose_clip

            #Read just the scan branch
            # if os.path.isfile(os.path.join(pose_dir, "valid_poses_colmap_by_scan_filtered.npy")) and os.path.isfile(os.path.join(pose_dir, "poses_colmap_by_scan_filtered.npy")):
            #    print("Extracting from colmap computed from poses by scan branch")
            #    valid_pose = np.load(os.path.join(pose_dir, "valid_poses_colmap_by_scan_filtered.npy"))
            #    Ci_T_G = np.load(os.path.join(pose_dir, "poses_colmap_by_scan_filtered.npy"))
            #    return  Ci_T_G ,valid_pose
            
            # Ambas ramas
            if os.path.isfile(os.path.join(pose_dir, "valid_poses_colmap_by_scan_filtered.npy")) and os.path.isfile(os.path.join(pose_dir, "poses_colmap_by_scan_filtered.npy")) and not os.path.isfile(os.path.join(pose_dir, "valid_poses_colmap_filtered.npy")) and not os.path.isfile(os.path.join(pose_dir, "poses_colmap_filtered.npy")):
                print("Extracting from colmap computed from poses by scan")
                valid_pose = np.load(os.path.join(pose_dir, "valid_poses_colmap_by_scan_filtered.npy"))
                Ci_T_G = np.load(os.path.join(pose_dir, "poses_colmap_by_scan_filtered.npy"))
                return  Ci_T_G ,valid_pose 
            
            if os.path.isfile(os.path.join(pose_dir, "valid_poses_colmap_by_scan_filtered.npy")) and os.path.isfile(os.path.join(pose_dir, "poses_colmap_by_scan_filtered.npy")) and os.path.isfile(os.path.join(pose_dir, "valid_poses_colmap_filtered.npy")) and os.path.isfile(os.path.join(pose_dir, "poses_colmap_filtered.npy")):
                print("Extracting from colmap computed from poses")
                valid_pose_scan = np.load(os.path.join(pose_dir, "valid_poses_colmap_by_scan_filtered.npy"))
                Ci_T_G_scan = np.load(os.path.join(pose_dir, "poses_colmap_by_scan_filtered.npy"))
                valid_pose_clip = np.load(os.path.join(pose_dir, "valid_poses_colmap_filtered.npy"))
                Ci_T_G_clip = np.load(os.path.join(pose_dir, "poses_colmap_filtered.npy"))
                for i in range(len(valid_pose_clip)):
                    if not valid_pose_clip[i]:
                        if valid_pose_scan[i]:
                            valid_pose_clip[i] = True
                            Ci_T_G_clip[i] = Ci_T_G_scan[i]
                return  Ci_T_G_clip ,valid_pose_clip 
            
        print("Extracting from pnp")
        if not os.path.isfile(os.path.join(pose_dir,
                                           'cameras_pnp_triangulation.npy')):
            return None

        original_image_ids = np.arange(0,len(fnmatch.filter(os.listdir(os.path.join(dirname,
                                                                                     'color')),
                                                             '*.jpg')))

        valid_pose = np.load(os.path.join(pose_dir, 'good_pose_reprojection.npy'))
        C_T_G = np.load(os.path.join(pose_dir, 'cameras_pnp_triangulation.npy'))
        
        # Filtrar poses
        #valid_pose = open3d_outlier_removal(C_T_G, valid_pose)
        Ci_T_G = np.zeros((len(original_image_ids), 4, 4))
        k = 0
        for i in range(len(original_image_ids)):
            if valid_pose[i]:
                Ci_T_G[k] = np.concatenate((C_T_G[i], np.array([[0., 0., 0., 1.]])), axis=0)
                k += 1
            else:
                Ci_T_G[k] = np.eye(4)
                Ci_T_G[k][2, 3] = 100
                k += 1
        #print(sum(valid_pose))
        return Ci_T_G, valid_pose

    def load_3d_annotation(self, data: Dict):
        # use Ego4D-3D-Annotation API
        box = BoundingBox()
        box.load(data)
        return box.center


    def create_traj_azure(self, output_traj, K, Ci_T_G=None):

        d = json.load(open('../camera_pose_estimation/Visualization/camera_trajectory.json', 'r'))
        dp0 = d['parameters'][0]
        dp0['intrinsic']['width'] = int((K[6]+0.5)*2)
        dp0['intrinsic']['height'] = int((K[7]+0.5)*2)
        dp0['intrinsic']['intrinsic_matrix'] = K.tolist()
        dp0['extrinsic'] = []
        x = []


        if Ci_T_G is not None:
            for i in range(Ci_T_G.shape[0]):
                temp = dp0.copy()

                # E = np.linalg.inv(G_T_Ci[i])
                E = Ci_T_G[i]

                E_v = np.concatenate([E[:, i] for i in range(4)], axis=0)
                temp['extrinsic'] = E_v.tolist()
                x.append(temp)

        d['parameters'] = x
        with open(output_traj, 'w') as f:
            json.dump(d, f)



    def read_pfm(self, path):
        import sys
        import re
        import numpy as np
        import cv2
        import torch

        from PIL import Image
        """Read pfm file.
        Args:
            path (str): path to file
        Returns:
            tuple: (data, scale)
        """
        with open(path, "rb") as file:

            color = None
            width = None
            height = None
            scale = None
            endian = None

            header = file.readline().rstrip()
            if header.decode("ascii") == "PF":
                color = True
            elif header.decode("ascii") == "Pf":
                color = False
            else:
                raise Exception("Not a PFM file: " + path)

            dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
            if dim_match:
                width, height = list(map(int, dim_match.groups()))
            else:
                raise Exception("Malformed PFM header.")

            scale = float(file.readline().decode("ascii").rstrip())
            if scale < 0:
                # little-endian
                endian = "<"
                scale = -scale
            else:
                # big-endian
                endian = ">"

            data = np.fromfile(file, endian + "f")
            shape = (height, width, 3) if color else (height, width)

            data = np.reshape(data, shape)
            data = np.flipud(data)

            return data, scale


