from bounding_box import BoundingBox
import json
import argparse
import os
import sys
import numpy as np

sys.path.append('../../VQ3D/API/')
from get_query_3d_ground_truth import VisualQuery3DGroundTruth
import open3d as o3d

Rz_90 = np.array([[np.cos(-np.pi/2), -np.sin(-np.pi/2), 0, 0],
                  [np.sin(-np.pi/2),  np.cos(-np.pi/2), 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1],
                 ])

def is_point_in_box(point, vertices):
    """
    Determines if a point is inside a 3D box defined by its vertices
    :param point: point to be checked (x, y, z)
    :param vertices: vertices of the box as an array of shape (8, 3) where each row is a vertex (x, y, z)
    :return: True if the point is inside the box, False otherwise
    """
    # calculate the max and min coordinates for each axis
    min_x = min(vertices[:, 0])
    max_x = max(vertices[:, 0])
    min_y = min(vertices[:, 1])
    max_y = max(vertices[:, 1])
    min_z = min(vertices[:, 2])
    max_z = max(vertices[:, 2])
    
    # check if the point is inside the box
    if (min_x <= point[0] <= max_x and
        min_y <= point[1] <= max_y and
        min_z <= point[2] <= max_z):
        return True
    else:
        return False

def extract_off_annot(args):
    vq3d_annot =  json.load(open(args.annot_path, "rb"))
    helper = VisualQuery3DGroundTruth()
    for video in vq3d_annot["videos"]:
        scan_uid = video["scan_uid"]
        for clip in video["clips"]:
            clip_uid = clip['clip_uid']
            for ai, annot in enumerate(clip['annotations']):
                if not annot: continue
                for qset_id, qset in annot['query_sets'].items():
                    object_title = qset["object_title"]
                    annotation_1_dict = qset["3d_annotation_1"]
                    annotation_2_dict = qset["3d_annotation_2"]
                    annotation_1 =  BoundingBox(annotation_1_dict)
                    annotation_2 =  BoundingBox(annotation_2_dict)
                    annotation_1.save_off(os.path.join(args.otuput_dir,"{}_{}_{}_{}_annot1.off".format(clip_uid,ai,qset_id, scan_uid)), 1)
                    annotation_2.save_off(os.path.join(args.otuput_dir,"{}_{}_{}_{}_annot2.off".format(clip_uid,ai,qset_id, scan_uid)), 2)
                    point_cloud = o3d.geometry.PointCloud()
                    point_clouds=[]
                    for w in [annotation_1, annotation_2]:
                        v0,v1,v2,v3,v4,v5,v6,v7 = w.build_box()
                        vertices = np.array([v0[:3],v1[:3],v2[:3],v3[:3],v4[:3],v5[:3],v6[:3],v7[:3]])
                        breakpoint()
                        c = np.append(w.center, 1.)
                        c = np.matmul(Rz_90, c)
                        c = c[:3] / c[3]
                        if not is_point_in_box(c, vertices):
                            print(clip_uid,ai,qset_id, scan_uid)
                        point_clouds.append(c)
                    point_cloud.points = o3d.utility.Vector3dVector(point_clouds)
                    o3d.io.write_point_cloud(os.path.join(args.otuput_dir,"{}_{}_{}_{}_centers.ply".format(clip_uid,ai,qset_id, scan_uid)), point_cloud)





if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        default='val'
    )
    parser.add_argument(
        "--clip_uid",
        type=str,
        default='6c641082-044e-46a7-ad5f-85568119e09e'
    )
    parser.add_argument(
        "--otuput_dir",
        type=str,
        default='/media/SSD5/ego4d/dataset/3d/v1/visualizations_annot'
    )
    parser.add_argument(
        "--annot_path",
        type=str,
        default='/media/SSD5/ego4d/last_annotations/3d/vq3d_val.json'
    )
    args = parser.parse_args()
    extract_off_annot(args)
    