import numpy as np
import tensorflow as tf
import pickle
import json
from typing import Union, Tuple, Dict
from pyntcloud import PyntCloud
import pandas as pd
import os
import tqdm
import open3d as o3d
import random
import glob
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def translate_camera_extrinsics(p, new_center):
    s = 1.0 # Scale is the same
    r = to_homogeneous(np.identity(3)) # Identity matrix
    s = scale([s, s, s])
    center = camera_center_from_extrinsics(p[:3,:]) # translation vector
    t = new_center - center
    t = translate(tf.squeeze(t))

    T_proc = t @ r @s

    p_modified = apply_procrustes_to_extrinsics(p, T_proc)

    return p_modified

def load_vq3d_annotation(filename):
    output = {}
    data = json.load(open(filename, 'r'))
    for video in data['videos']:
        scan_uid = video['scan_uid']
        for clip in video['clips']:
            output[clip['clip_uid']] = scan_uid
    return output

def constraint_inside(points, mesh):
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)
    new_points = {}
    for i, tupl in enumerate(zip(points.keys(), points.values())):
        key, value = tupl[0], tupl[1]
        center = camera_center_from_extrinsics(value)
        query_point = o3d.core.Tensor([center], dtype=o3d.core.Dtype.Float32)
        occupancy = scene.compute_occupancy(query_point).item()
        if bool(occupancy):
            new_points[key]=value
    print("Original: {}, Filtered: {}".format(len(points), len(new_points)))
    return new_points

def constraint_inside_and_relocate(extrinsics, mesh, scene):
    center = camera_center_from_extrinsics(extrinsics)
    query_point = o3d.core.Tensor([center], dtype=o3d.core.Dtype.Float32)
    occupancy = scene.compute_occupancy(query_point).item()
    extrinsics =  tf.concat((extrinsics, [[0, 0, 0, 1]]), axis=0)
    if not bool(occupancy):
        # Compute the closest vertex to the scan from the point
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector([center])
        distances = o3d.geometry.PointCloud(mesh.vertices).compute_point_cloud_distance(pcd)
        closest_vertex_index = np.argmin(distances)
        closest_vertex = mesh.vertices[closest_vertex_index]
        #translate the extrinsics
        extrinsics =  translate_camera_extrinsics(extrinsics, closest_vertex)
    return extrinsics

def load_vq3d_annotation(filename):
    output = {}
    data = json.load(open(filename, 'r'))
    for video in data['videos']:
        scan_uid = video['scan_uid']
        for clip in video['clips']:
            output[clip['clip_uid']] = scan_uid
    return output

def open3d_outlier_removal(points, nb_neighbors=15, std_ratio=3):
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

def create_lines_between_points(list1, list2, path_name):
    list1 = np.array(list1)
    list2 = np.array(list2)
    assert len(list1) == len(list2)
    line_sets = []
    for i in range(list1.shape[1]):
        # Create line segment between the two points
        points = [list1[:,i], list2[:,i]]
        lines = [[0, 1]]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_sets.append(line_set)
    # Combine all line sets into one
    combined_line_set = line_sets[0]
    for i in range(1, len(line_sets)):
        combined_line_set += line_sets[i]
    # Save combined line set as PLY file
    o3d.io.write_line_set(path_name+"_lines.ply", combined_line_set)

def visualize_points(list1, list2, path_name, clip_uid):
    list1 = np.array(list1)
    list2 = np.array(list2)
    assert len(list1)==len(list2)
    point_clouds = []
    colors = []
    lines = []  # new list to store line indices
    for i in range(list1.shape[1]):
        point_clouds.append(list1[:,i])
        point_clouds.append(list2[:,i])
        color = [random.uniform(0, 1) for _ in range(3)]
        colors.append(color)
        colors.append(color)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_clouds)
    pcd.colors = o3d.utility.Vector3dVector(colors)    
    o3d.io.write_point_cloud(path_name+"_point_cloud.ply", pcd)
    # cargar mesh
    scan_name = clip_uid_to_scan_uid[clip_uid]
    scan_uid = scan_name_to_uid[scan_name]
    root = "/media/SSD0/mcescobar/episodic-memory/VQ3D/data/v1/3d/3d_scans/{}".format(scan_uid)
    mesh = o3d.io.read_triangle_mesh(os.path.join(root, glob.glob(root+"/*.obj")[0]))
    o3d.io.write_triangle_mesh(path_name+"_mesh.ply", mesh, write_ascii=False)
    create_lines_between_points(list1, list2, path_name)

def to_rotation_matrix(r: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
    """Enforces a rotation matrix form of the input.

    Args:
    r: N x N input array. Can be a numpy array or a tf.Tensor

    Returns:
    Orthonormal rotation tensor N x N.
    """
    if r.shape[0] != r.shape[1]:
        raise ValueError('Rotation matrix must be rectangular')

    r = tf.convert_to_tensor(r)

    _, u, v = tf.linalg.svd(r)

    # Handle case where determinant of rotation matrix is negative.
    r = tf.matmul(v, tf.transpose(u))
    correction = tf.cond(
        tf.linalg.det(r) < 0, lambda: tf.linalg.diag([1, 1, -1]),
        lambda: tf.linalg.diag([1, 1, 1]))

    v = tf.matmul(v, tf.cast(correction, v.dtype))
    r = tf.matmul(v, tf.transpose(u))

    return r

def fit_rigid_transform(
    p: Union[tf.Tensor, np.ndarray],
    q: Union[tf.Tensor, np.ndarray]) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Fit rigid transformation tensors for rotation, translation, and scale.

    Implementation follows:
    1. https://ieeexplore.ieee.org/document/4767965
    2. https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
    See scipy.spatial.procrustes for the black-box method.

    Args:
    p: 3 x N points of source. Can be a numpy array or a tf.Tensor
    q: 3 x N points of destination. Can be a numpy array or a tf.Tensor

    Returns:
    Rotation tensor r, translation tensor t, and scale,
    such that q = tf.matmul(r, scale * p) + t
    """
    
    p = tf.convert_to_tensor(p, dtype=tf.float32)
    q = tf.convert_to_tensor(q, dtype=tf.float32)

    if p.shape[0] != 3 or q.shape[0] != 3:
        print('3D points are required as input.', p.shape[0], q.shape[0])
        return None, None, None
    if p.shape[1] < 3 or q.shape[1] < 3:
        print('3 or more 3D points are required.',p.shape[1], q.shape[1] )
        return None, None, None 

    # Compute source and destination centroids.
    c_p = tf.reduce_mean(p, axis=1, keepdims=True)
    c_q = tf.reduce_mean(q, axis=1, keepdims=True)

    # Compute average distance to centroid for scale
    dist_p = tf.norm(p - tf.tile(c_p, (1, p.shape[1])), axis=0)
    dist_q = tf.norm(q - tf.tile(c_q, (1, q.shape[1])), axis=0)
    scale = tf.divide(
        tf.reduce_mean(dist_q), tf.maximum(tf.reduce_mean(dist_p), 1e-7))

    # Bring source and destination points to origin
    p_norm = tf.multiply(scale, p - c_p)
    q_norm = q - c_q

    # Compute r and t
    h = tf.matmul(p_norm, tf.transpose(q_norm))
    r = to_rotation_matrix(h)
    t = c_q - tf.matmul(r, tf.multiply(scale, c_p))

    return r, t, scale

def camera_center_from_extrinsics(p: tf.Tensor):
    """Computes camera center from extrinsics. p is the 3 x 4 extrinsics."""
    r = p[:3, :3]
    t = p[:, -1][..., tf.newaxis]
    return tf.squeeze(-tf.linalg.inv(r) @ t, axis=1)

def gather_mutual_centers(extr_dict_source: Dict[str, tf.Tensor],
                        extr_dict_destination: Dict[str, tf.Tensor]):
    """Returns frame names, camera centers on source, and centers on destination.

    Args:
    extr_dict_source: {frame_name: extrinsics (3 x 4)} mapping for source (COLMAP).
    extr_dict_destination: {frame_name: extrinsics (3 x 4)} mapping for destination (scan).
    """
    print("Colmap: {} / PnP: {}".format(len(extr_dict_source), len(extr_dict_destination)))
    frame_names = sorted(extr_dict_destination)
    centers_source = []
    centers_destination = []
    for frame_name in frame_names:
        if frame_name not in extr_dict_source:
            continue
        p_source = extr_dict_source[frame_name]
        centers_source.append(camera_center_from_extrinsics(p_source))
        p_destination = extr_dict_destination[frame_name]
        centers_destination.append(camera_center_from_extrinsics(p_destination))
    if len(centers_source)==0 or len(centers_destination)==0:
        return frame_names, centers_source, centers_destination
    centers_source =  tf.convert_to_tensor(centers_source, dtype=tf.float32)
    centers_source = tf.transpose(centers_source,[1,0])
    centers_destination = tf.transpose(tf.convert_to_tensor(centers_destination, dtype=tf.float32), [1,0])
    
    return frame_names, centers_source, centers_destination

def translate(t):
    diag =  np.diag([1.0,1.0,1.0,1.0])
    diag[:3,3] = t
    return tf.convert_to_tensor(diag, dtype=tf.float32)

def scale(s):
    s.append(1)
    return tf.convert_to_tensor(np.diag(s), dtype=tf.float32)

def to_homogeneous(r):
    rta = np.zeros((4,4))
    rta[0:3,0:3] =  r
    rta[-1,-1] = 1
    return rta

def get_procrustes_transform(p1, p2):
    """Gets the 7-DOF transformation from p1 to p2."""
    r, t, s = fit_rigid_transform(p1, p2)
    if r is None or t is None or s is None: 
        return None
    r = to_homogeneous(r) # creates a 4x4 matrix where [0:3, 0:3] is the rotation matrix.
    s = scale([s, s, s]) # Creates a diag([s, s, s, 1]) matrix.
    t = translate(tf.squeeze(t)) # creates a 4 x 4 matrix where [:, 3] is the translation vector.
    return t @ r @s

def transform_points(centers_colmap, T_proc):
    centers_colmap = np.append(centers_colmap, np.ones((1,centers_colmap.shape[1])), axis=0)
    return T_proc @ centers_colmap
def apply_procrustes_to_extrinsics(extrinsic, trans):
    """Transform extrinsics with procrustes."""
    tmp = tf.cast(extrinsic, tf.float32) @ tf.linalg.inv(trans)
    tmp = tmp / tf.reduce_mean(tf.norm(tmp[:3, :3], axis=0))
    return tf.concat((tmp[:3, :], [[0, 0, 0, 1]]), axis=0)

def export_to_pyl(points, colmap, filename):
    if type(points)==np.ndarray:
        pass
    else:
        points =  np.asarray(points).transpose()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    filename = f"{filename}.ply"
    o3d.io.write_point_cloud(filename, pcd)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        default = "val",
        help="val/test",
    )
    parser.add_argument(
        "--visualize",
        action='store_true',
        help="Visualize points",
    )
    parser.add_argument(
        "--filter",
        action='store_true',
        help="Visualize points",
    )

    parser.add_argument(
        "--constrain",
        action='store_true',
        help="Visualize points",
    )
    parser.add_argument(
        "--path_to_clips_frames",
        type=str,
        default="/media/SSD5/ego4d/dataset/3d/v1/clips_5fps_frames"
        help="Input folder with the clips.",
    )
    parser.add_argument(
        "--path_to_scans",
        type=str,
        default="/media/SSD0/mcescobar/episodic-memory/VQ3D/data/v1/3d/3d_scans/"
        help="path to the folder with the 3d scans information",
    )
    args = parser.parse_args()

    scan_name_to_uid = {
    'unict_Scooter mechanic_31': 'unict_3dscan_001',
    'unict_Baker_32': 'unict_3dscan_002',
    'unict_Carpenter_33': 'unict_3dscan_003',
    'unict_Bike mechanic_34': 'unict_3dscan_004',
    }
    name_vq3d_annotation =  '/media/SSD3/ego4d/dataset/3d/v1/annotations/vq3d_val.json' if args.split == "val" else "/media/SSD0/mcescobar/episodic-memory/VQ3D/data/v1/annotations/vq3d_test_unannotated.json" 
    clip_uid_to_scan_uid = load_vq3d_annotation(name_vq3d_annotation)


    if args.visualize:
        if args.filter:
            visualization_folder = "visualizations_filtered" 
        if args.constrain:
            visualization_folder = "visualizations_constrain"
        else:
            visualization_folder="visualizations"
        if not os.path.exists(visualization_folder):
            os.makedirs(os.path.join(visualization_folder,"common_by_clip"))
        if not os.path.exists(os.path.join(visualization_folder,"common_by_clip")):
            os.makedirs(os.path.join(visualization_folder,"common_by_clip"))
        if not os.path.exists(os.path.join(visualization_folder,"common_by_clip_colors")):
            os.makedirs(os.path.join(visualization_folder,"common_by_clip_colors"))
        if not os.path.exists(os.path.join(visualization_folder,"all_by_clip")):
            os.makedirs(os.path.join(visualization_folder,"all_by_clip"))
        if not os.path.exists(os.path.join(visualization_folder,"colmap_relative_by_clip")):
            os.makedirs(os.path.join(visualization_folder,"colmap_relative_by_clip"))
    
    colmap_name = "colmap_ext_val.pkl" if args.split=="val" else "colmap_ext_test.pkl"
    pnp_name =  "pnp_ext_val.pkl" if args.split=="val" else "pnp_ext_test.pkl"
    extr_dict_colmap_total = pickle.load(open(colmap_name, "rb"))
    extr_dict_pnp_total = pickle.load(open(pnp_name, "rb"))

    path_to_clips_frames = args.path_to_clips_frames
    
    root_dir ="/media/SSD5/ego4d/dataset/3d/v1/clips_camera_poses_5fps" if args.split=="val" else "/media/SSD5/ego4d/dataset/3d/v1/clips_camera_poses_5fps_test"
    
    valid_clips = 0
    total_clips = 0
    total_frames = 0
    valid_frames = 0

    for clip in extr_dict_colmap_total:
        # eliminamos las poses calculadas con anterioridad
        if os.path.exists(os.path.join(root_dir, clip, "egovideo",'superglue_track', 'poses', "valid_poses_colmap_filtered.npy")):
            os.remove(os.path.join(root_dir, clip, "egovideo",'superglue_track', 'poses', "valid_poses_colmap_filtered.npy"))
        if os.path.exists(os.path.join(root_dir, clip, "egovideo",'superglue_track', 'poses', "poses_colmap_filtered.npy")):
            os.remove(os.path.join(root_dir, clip, "egovideo",'superglue_track', 'poses', "poses_colmap_filtered.npy"))

        total_clips+=1
        total_frames += len(os.listdir(os.path.join(path_to_clips_frames, clip)))
        extr_dict_colmap = extr_dict_colmap_total[clip]
        extr_dict_pnp = extr_dict_pnp_total[clip]

        if args.filter:
            #filtrar outliers
            extr_dict_colmap=open3d_outlier_removal(extr_dict_colmap)
            extr_dict_pnp=open3d_outlier_removal(extr_dict_pnp)

        if args.constrain and len(extr_dict_pnp)>0:
            scan_name = clip_uid_to_scan_uid[clip]
            scan_uid = scan_name_to_uid[scan_name]
            root = os.path.join(args.path_to_scans, scan_uid)
            # Load the scan from the .obj file
            mesh = o3d.io.read_triangle_mesh(os.path.join(root, glob.glob(root+"/*.obj")[0]))
            mesh_legacy = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            scene = o3d.t.geometry.RaycastingScene()
            _ = scene.add_triangles(mesh_legacy)
            # Filter pnp points that are not inside the scan
            extr_dict_pnp = constraint_inside(extr_dict_pnp, mesh)
            extr_dict_colmap=open3d_outlier_removal(extr_dict_colmap)

        if args.visualize:
            datos = []
            for frame in extr_dict_colmap:
                center = camera_center_from_extrinsics(extr_dict_colmap[frame])
                datos.append(center)
            datos = np.array(datos)
            export_to_pyl(datos, True, os.path.join(visualization_folder,"colmap_relative_by_clip",clip+".ply"))

        if len(extr_dict_pnp)==0:
            continue

        # Compute transformations of camera centers with procrustes.
        # We assume that the mapping of {frame_name: extrinsics (3 x 4)} is given both for colmap and the scan.
        frame_names, centers_colmap, centers_pnp = gather_mutual_centers(extr_dict_colmap, extr_dict_pnp)
        if len(centers_colmap)==0 or len(centers_pnp)==0:
            continue

        T_proc = get_procrustes_transform(centers_colmap, centers_pnp)
        if T_proc is None:
            continue

        valid_clips += 1
        valid_frames += len(extr_dict_colmap)
        centers_colmap_onscan = transform_points(centers_colmap, T_proc)

        if args.visualize:
            assert centers_colmap_onscan.shape[1]==centers_pnp.shape[1] 
            export_to_pyl(centers_colmap_onscan[0:3,:], True, os.path.join(visualization_folder,"common_by_clip",clip+"_colmap"))
            export_to_pyl(centers_pnp, False, os.path.join(visualization_folder,"common_by_clip",clip+"_pnp"))
            visualize_points(centers_colmap_onscan[0:3,:], centers_pnp, os.path.join(visualization_folder,"common_by_clip_colors",clip), clip)
        
        # Apply procrustes to extrinsic COLMAP results.
        extr_colmap_onscan = []
        valid_poses = []

        for i in tqdm.tqdm(range(len(os.listdir(os.path.join(path_to_clips_frames, clip))))):
            if 'color_%07d.jpg'%i in extr_dict_colmap:
                if args.constrain:
                    pose = constraint_inside_and_relocate(extr_dict_colmap['color_%07d.jpg'%i] ,mesh, scene)
                    extr_colmap_onscan.append(pose)
                    if not(pose.shape[0]==4 and pose.shape[1]==4):
                        continue
                else:
                    extr_colmap = tf.concat((extr_dict_colmap['color_%07d.jpg'%i], [[0, 0, 0, 1]]), axis=0)
                    extr_colmap_onscan.append(apply_procrustes_to_extrinsics(extr_colmap, T_proc))
                valid_poses.append(True)
            else:
                valid_poses.append(False)
                extr_colmap_onscan.append(np.array([[  1.,   0.,   0.,   0.],
                                                    [  0.,   1.,   0.,   0.],
                                                    [  0.,   0.,   1., 100.],
                                                    [  0.,   0.,   0.,   1.]]))
        print(sum(valid_poses),"/",len(valid_poses))
        
        path_to_save_valid_poses = os.path.join(root_dir, clip, "egovideo",'superglue_track', 'poses', "valid_poses_colmap_filtered.npy")
        path_to_save_poses = os.path.join(root_dir, clip, "egovideo",'superglue_track', 'poses', "poses_colmap_filtered.npy")
        np.save(path_to_save_valid_poses, np.array(valid_poses))
        np.save(path_to_save_poses, np.array(extr_colmap_onscan))
        
        if args.visualize:
            # Create ply for all the colmap valid poses on scan:
            datos = []
            for valid, pose in zip(valid_poses, extr_colmap_onscan):
                if valid:
                    center  = camera_center_from_extrinsics(pose[0:3,:])
                    datos.append(center)
            datos = np.array(datos)
            export_to_pyl(datos, True, os.path.join(visualization_folder,"all_by_clip",clip+"_colmap"))

            datos = []
            for frame in extr_dict_pnp:
                center = camera_center_from_extrinsics(extr_dict_pnp[frame])
                datos.append(center)
            datos = np.array(datos)
            export_to_pyl(datos, True, os.path.join(visualization_folder,"all_by_clip",clip+"_pnp.ply"))
                


    print("Clips: {}/{} ({}), Frames {}/{} ({})".format(valid_clips, total_clips, valid_clips*100/total_clips, valid_frames, total_frames, valid_frames*100/total_frames))