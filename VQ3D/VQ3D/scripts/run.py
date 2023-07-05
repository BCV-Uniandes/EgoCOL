import os
import sys
import json
#import h5py
import torch
import glob
import argparse
import numpy as np
from pyntcloud import PyntCloud
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
import open3d as o3d

sys.path.append('API/')
from get_query_3d_ground_truth import VisualQuery3DGroundTruth

sys.path.append('../annotation_API/API/')
from bounding_box import BoundingBox

sys.path.append('../../colmap')
from transform_ext import constraint_inside_and_relocate

def camera_center_from_extrinsics(p: torch.Tensor):
    """Computes camera center from extrinsics. p is the 3 x 4 extrinsics."""
    r = p[:3, :3]
    t = p[:, -1].reshape(-1, 1)
    return np.squeeze(-np.linalg.inv(r) @ t, axis=1)

def constraint_inside_and_relocate_point(center, mesh, scene):
    query_point = o3d.core.Tensor([center], dtype=o3d.core.Dtype.Float32)
    occupancy = scene.compute_occupancy(query_point).item()
    if not bool(occupancy):
        # Compute the closest vertex to the scan from the point
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector([center])
        distances = o3d.geometry.PointCloud(mesh.vertices).compute_point_cloud_distance(pcd)
        closest_vertex_index = np.argmin(distances)
        closest_vertex = mesh.vertices[closest_vertex_index]
        return closest_vertex
    else:
        return center

def load_vq3d_annotation(filename):
    output = {}
    data = json.load(open(filename, 'r'))
    for video in data['videos']:
        scan_uid = video['scan_uid']
        for clip in video['clips']:
            output[clip['clip_uid']] = scan_uid
    return output

def save_pred_gt(gt, pred, filename):
    point_clouds = [gt, pred]
    colors = [[0,1,0],[0,0,1]] # Verde el gt y azul el pred
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_clouds)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    root = "/media/SSD0/mcescobar/episodic-memory/VQ3D/data/vq3d_results/visualizations_centers"
    filename = os.path.join(root,filename)
    o3d.io.write_point_cloud(filename, pcd)

def scale_im_height(image, H):
    im_H, im_W = image.shape[:2]
    W = int(1.0 * H * im_W / im_H)
    return cv2.resize(image, (W, H))

def _get_box(annot_box):
    x, y, w, h = annot_box["x"], annot_box["y"], annot_box["width"], annot_box["height"]
    return (int(x), int(y), int(x + w), int(y + h))

def parse_VQ2D_queries(filename: str) -> Dict:
    output = {}
    data = json.load(open(filename, 'r'))
    for video in data['videos']:
        video_uid = video['video_uid']
        if video_uid not in output:
            output[video_uid]={}
        for clip in video['clips']:
            clip_uid = clip['clip_uid']
            if clip_uid not in output[video_uid]:
                output[video_uid][clip_uid]={}
            for ai, annot in enumerate(clip['annotations']):
                if ai not in output[video_uid][clip_uid]:
                    output[video_uid][clip_uid][ai]={}
                for qset_id, qset in annot['query_sets'].items():
                    if not qset["is_valid"]:
                        continue
                    output[video_uid][clip_uid][ai][qset_id] = {
                        "query_frame": qset["query_frame"],
                        "object_title": qset["object_title"],
                        "visual_crop": qset["visual_crop"],
                        "response_track": qset["response_track"],
                    }
    return output

def parse_VQ2D_predictions(filename: str) -> Dict:
    output = {}
    data = json.load(open(filename, 'r'))['predictions']
    for i in range(len(data['dataset_uids'])):
        dataset_uid = data['dataset_uids'][i]
        output[dataset_uid] = {'pred': data['predicted_response_track'][0][i],
                               'gt': data['ground_truth_response_track']}
    return output


def parse_VQ2D_mapper(filename: str) -> Dict:
    data = json.load(open(filename,'r'))
    output = {}
    for i in range(len(data)):
        dataset_uid = data[i]['dataset_uid']
        video_uid = data[i]['metadata']['video_uid']
        clip_uid = data[i]['clip_uid']
        query_set = data[i]['query_set']
        query_frame = data[i]['query_frame']
        object_title = data[i]['object_title']
        visual_crop = data[i]['visual_crop']
        if video_uid not in output:
            output[video_uid]={}
        if clip_uid not in output[video_uid]:
            output[video_uid][clip_uid]={}
        if query_set not in output[video_uid][clip_uid]:
            output[video_uid][clip_uid][query_set]={}
        if query_frame not in output[video_uid][clip_uid][query_set]:
            output[video_uid][clip_uid][query_set][query_frame]=[]

        output[video_uid][clip_uid][query_set][query_frame].append(
            {'dataset_uid':dataset_uid,
             'object_title':object_title,
             'visual_crop':visual_crop,
            }
        )
    return output

def dict_map_f(root_dir, clip_uid):
    path = os.path.join(root_dir, clip_uid,"egovideo", "color")
    images = sorted(os.listdir(path))
    dict_map = {}
    for indx,image in enumerate(images):
        dict_map[int(image.split("_")[1].replace(".jpg",""))] = indx
    return dict_map 

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

def visualize_points(list1, list2, path_name):
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
    create_lines_between_points(list1, list2, path_name)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default='data/clips_from_videos_camera_poses/',
        help="Camera pose folder"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default='data/vq3d_results/siam_rcnn_residual_kys_val.json',
        help="Camera pose folder"
    )
    parser.add_argument(
        "--vq2d_results",
        type=str,
        default='data/vq2d_results/siam_rcnn_residual_kys_val.json',
        help="filename for the VQ2D results"
    )
    parser.add_argument(
        "--vq2d_annot",
        type=str,
        default='data/val_annot.json',
        help="VQ2D mapping queries"
    )
    parser.add_argument(
        "--vq3d_queries",
        type=str,
        default='data/vq3d_val_wgt.json',
        help="VQ3D query file"
    )
    parser.add_argument(
        "--vq2d_queries",
        type=str,
        default='data/vq_val.json',
        help="VQ3D query file"
    )
    parser.add_argument(
        "--use_gt",
        action='store_true'
    )
    parser.add_argument(
        "--use_depth_from_scan",
        action='store_true'
    )
    parser.add_argument(
        '--check_colmap',
        action='store_true'
    )

    parser.add_argument(
        '--constrain',
        action='store_true'
    )

    parser.add_argument(
        '--baseline_center',
        action='store_true'
    )

    parser.add_argument(
        "--poses_as_pred",
        action='store_true'
    )

    args = parser.parse_args()

    root_dir = args.input_dir

    output_filename = args.output_filename
    check_colmap =  args.check_colmap
    # Visual Query 3D queries
    vq3d_queries = json.load(open(args.vq3d_queries, 'r'))

    # Visual Query 2D results
    vq2d_queries = parse_VQ2D_queries(args.vq2d_queries)
    vq2d_pred = parse_VQ2D_predictions(args.vq2d_results)
    vq2d_mapping = parse_VQ2D_mapper(args.vq2d_annot)

    scan_name_to_uid = {
    'unict_Scooter mechanic_31': 'unict_3dscan_001',
    'unict_Baker_32': 'unict_3dscan_002',
    'unict_Carpenter_33': 'unict_3dscan_003',
    'unict_Bike mechanic_34': 'unict_3dscan_004',
    }


    # Load mapping VQ2D to VQ3D queries/annotations
    if 'val' in args.vq2d_queries:
        split='val'
    elif 'train' in args.vq2d_queries:
        split='train'
    elif 'test' in args.vq2d_queries:
        split='test'
    else:
        raise ValueError
    query_matching_filename=f'data/mapping_vq2d_to_vq3d_queries_annotations_{split}.json'
    query_matching = json.load(open(query_matching_filename, 'r'))

    name_vq3d_annotation =  '/media/SSD3/ego4d/dataset/3d/v1/annotations/vq3d_val.json' if split == "val" else "/media/SSD0/mcescobar/episodic-memory/VQ3D/data/v1/annotations/vq3d_test_unannotated.json" 
    clip_uid_to_scan_uid = load_vq3d_annotation(name_vq3d_annotation)

    depth_list = []
    annot_centers_list = []
    pred_centers_list = []

    helper = VisualQuery3DGroundTruth()
    response_track_valid = 0
    cpt_valid_queries = 0
    for video in vq3d_queries['videos']:
        video_uid = video['video_uid']
        for clip in video['clips']:
            clip_uid = clip['clip_uid']
            if args.constrain:
                scan_name = clip_uid_to_scan_uid[clip_uid]
                scan_uid = scan_name_to_uid[scan_name]
                root = "/media/SSD0/mcescobar/episodic-memory/VQ3D/data/v1/3d/3d_scans/{}".format(scan_uid)
                # Load the scan from the .obj file
                mesh = o3d.io.read_triangle_mesh(os.path.join(root, glob.glob(root+"/*.obj")[0]))
                mesh_legacy = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
                scene = o3d.t.geometry.RaycastingScene()
                _ = scene.add_triangles(mesh_legacy)

            dict_map = dict_map_f(root_dir, clip_uid)
            for ai, annot in enumerate(clip['annotations']):
                if not annot: continue
                for qset_id, qset in annot['query_sets'].items():

                    mapping_ai=query_matching[video_uid][clip_uid][str(ai)][qset_id]['ai']
                    mapping_qset_id=query_matching[video_uid][clip_uid][str(ai)][qset_id]['qset_id']

                    object_title=qset['object_title']
                    assert qset['object_title']==vq2d_queries[video_uid][clip_uid][mapping_ai][mapping_qset_id]['object_title']
                    query_frame=vq2d_queries[video_uid][clip_uid][mapping_ai][mapping_qset_id]['query_frame']
                    oW=vq2d_queries[video_uid][clip_uid][mapping_ai][mapping_qset_id]["visual_crop"]["original_width"]
                    oH=vq2d_queries[video_uid][clip_uid][mapping_ai][mapping_qset_id]["visual_crop"]["original_height"]

                    dataset_uid=vq2d_mapping[video_uid][clip_uid][qset_id][query_frame][0]['dataset_uid']
                    #dataset_uid=vq2d_mapping[video_uid][clip_uid][mapping_qset_id][query_frame][0]['dataset_uid']

                    # get intrinsics
                    camera_intrinsics = np.loadtxt(os.path.join(root_dir,
                                                                clip_uid,
                                                                'egovideo',
                                                                'fisheye_intrinsics.txt'))
                    W = camera_intrinsics[0]
                    H = camera_intrinsics[1]
                    f = camera_intrinsics[4]
                    k1 = camera_intrinsics[5]
                    k2 = camera_intrinsics[6]
                    cx = W/2.0
                    cy = H/2.0



                    # get poses
                    dirname = os.path.join(root_dir, clip_uid, 'egovideo')
                    
                    if not os.path.isdir(dirname):
                        print("NO PATH")
                        continue

                    poses = helper.load_pose(dirname, check_colmap)

                    if poses is None: 
                        #print("No poses")
                        #qset["pred_3d_vec_world"] = None
                        #qset["query_frame_pose"] = None
                        #qset["pred_3d_vec_query_frame"] = None
                        continue

                    T, valid_pose = poses
                    
                    frame_indices_valid = []
                    local_frame_indices = []
                    
                    # get RT frames with poses
                    if args.use_gt:
                        response_track=vq2d_queries[video_uid][clip_uid][mapping_ai][mapping_qset_id]["response_track"]
                        for i, frame in enumerate(response_track):
                            frame_index = frame['frame_number']

                            if (frame_index > -1) and (frame_index < len(valid_pose)):

                                box = _get_box(frame)
                                x1, y1, x2, y2 = box

                                if (x1<(W-1)) and (x2>1) and (y1<(H-1)) and (y2>1):
                                    # check if pose is valid
                                    if valid_pose[dict_map[frame_index]]:
                                        frame_indices_valid.append(frame_index)
                                        local_frame_indices.append(i)

                    else:
                        response_track = vq2d_pred[dataset_uid]['pred'][0]

                        frames = response_track['bboxes']

                        frame_indices = [x['fno'] for x in frames]

                        for i, frame_index in enumerate(frame_indices):

                            # check if frame index is valid
                            if (frame_index > -1):# and (frame_index < len(valid_pose)):

                                # check if box is within frame bound:
                                box = frames[i]
                                x1 = box['x1']
                                x2 = box['x2']
                                y1 = box['y1']
                                y2 = box['y2']

                                if (x1<(W-1)) and (x2>1) and (y1<(H-1)) and (y2>1):

                                    # check if pose is valid
                                    if valid_pose[dict_map[frame_index]]:
                                        frame_indices_valid.append(frame_index)
                                        local_frame_indices.append(i)

                    if len(frame_indices_valid) == 0:
                        
                        #qset["pred_3d_vec_world"] = None
                        #qset["query_frame_pose"] = None
                        #qset["pred_3d_vec_query_frame"] = None
                        continue

                    response_track_valid+=1

                    # get the last frame of the RT
                    j = np.argmax(frame_indices_valid)
                    frame_index_valid = frame_indices_valid[j]
                    local_frame_index = local_frame_indices[j]


                    # check if Query frame has pose
                    if valid_pose[dict_map[query_frame]]:
                        pose_Q = T[dict_map[query_frame]]
                        qset["query_frame_pose"] = pose_Q.tolist()
                    else:
                        pose_Q = None
                        qset["query_frame_pose"] = pose_Q


                    # get RT frame pose
                    pose = T[dict_map[frame_index_valid]]

                    #cpt_valid_queries+=1

                    # get depth
                    if args.use_depth_from_scan:
                        depth_dir = os.path.join(root_dir,
                                                clip_uid,
                                                'egovideo',
                                                'pose_visualization_depth_superglue'
                                                )
                        framename = 'render_%07d' % frame_index_valid
                        depth_filename = os.path.join(depth_dir,
                                                      framename+'.h5')
                        if os.path.isfile(depth_filename):
                            data = h5py.File(depth_filename)
                            depth = np.array(data['depth']) # in meters
                        else:
                            breakpoint()
                            print('missing predicted depth')
                            continue
                    else:
                        depth_dir = os.path.join(root_dir,
                                                clip_uid,
                                                'egovideo',
                                                'depth_DPT_predRT_CVPR_val'
                                                )
                        framename = 'color_%07d' % frame_index_valid
                        depth_filename = os.path.join(depth_dir,
                                                      framename+'.pfm')
                        if os.path.isfile(depth_filename):
                            data, scale = helper.read_pfm(depth_filename)
                        else:
                            print('missing predicted depth')
                            continue

                        depth = data/1000.0 # in meters

                    # resize depth
                    depth = torch.FloatTensor(depth)
                    depth = depth.unsqueeze(0).unsqueeze(0)
                    if not args.use_gt:
                        depth = torch.nn.functional.interpolate(depth,
                                                                size=(int(oH),
                                                                      int(oW)),
                                                                mode='bilinear',
                                                                align_corners=True)
                    else:
                        depth = torch.nn.functional.interpolate(depth,
                                                                size=(int(H), int(W)),
                                                                mode='bilinear',
                                                                align_corners=True)
                    depth = depth[0][0].cpu().numpy()

                    # select d
                    if args.use_gt:
                        box = _get_box(response_track[local_frame_index])
                        x1, y1, x2, y2 = box
                        if x1<0: x1=0
                        if y1<0: y1=0
                    else:
                        box = frames[local_frame_index]
                        x1 = box['x1']
                        x2 = box['x2']
                        y1 = box['y1']
                        y2 = box['y2']
                        if x1<0: x1=0
                        if y1<0: y1=0
                    
                    d = depth[y1:y2, x1:x2]

                    if d.size == 0:
                        print("NO DEPTH")
                        qset["pred_3d_vec_world"] = None
                        continue

                    d = np.median(d)
                    depth_list.append(d)

                    tx = (x1+x2)/2.0
                    ty = (y1+y2)/2.0

                    # vec in current frame:
                    z = d
                    x = z * (tx -cx -0.5)/f
                    y = z * (ty -cy -0.5)/f
                    vec = np.ones(4)
                    vec[0]=x
                    vec[1]=y
                    vec[2]=z
                    
                    # object center in world coord system
                    pred_t = np.matmul(pose, vec)
                    pred_t = pred_t / pred_t[3]
                    if args.poses_as_pred:
                        pred_t = camera_center_from_extrinsics(pose[:3,:])
                        pred_t = np.append(pred_t, 1.0)
                    if args.constrain:
                        pred_t = constraint_inside_and_relocate_point(pred_t[:3], mesh,scene)
                        pred_t = np.append(pred_t, 1.0)
                    if args.baseline_center:
                        scan_name = clip_uid_to_scan_uid[clip_uid]
                        scan_uid = scan_name_to_uid[scan_name]
                        root = "/media/SSD0/mcescobar/episodic-memory/VQ3D/data/v1/3d/3d_scans/{}".format(scan_uid)
                        # Load the scan from the .obj file
                        mesh = o3d.io.read_triangle_mesh(os.path.join(root, glob.glob(root+"/*.obj")[0]))
                        pred_t = mesh.get_center()
                        pred_t = np.append(pred_t, 1.0)


                    # object center in Query frame coord system
                    if pose_Q is not None:
                        vec = np.matmul(np.linalg.inv(pose_Q), pred_t)
                        vec = vec / vec[3]
                        vec = vec[:3]
                        qset['pred_3d_vec'] = vec.tolist()
                        l1 = np.linalg.norm(vec-qset['gt_3d_vec_1'])
                        l2 = np.linalg.norm(vec-qset['gt_3d_vec_2'])
                        print('L2 distance with annotation 1 and 2 in query frame coord system',
                              l1, ' ', l2)
                        #save pred y gt points to visualize
                        
                    else:
                        vec = None
                        #qset['pred_3d_vec_query_frame'] = None
                    cpt_valid_queries+=1
                    pred_t = pred_t[:3]
                    if(abs(np.sum(pred_t)))<15:
                        qset['pred_3d_vec_world'] = pred_t.tolist()
                    # visualize pred y gt
                    center1 = np.array(qset['gt_3d_vec_world_1'])
                    center2 = np.array(qset['gt_3d_vec_world_2'])
                    gt = (center1 + center2) / 2.0

                    annot_centers_list.append(gt)
                    pred = qset['pred_3d_vec_world']
                    pred_centers_list.append(pred)
                    #file_name = "{}_{}_{}_{}.ply".format(clip_uid, ai, qset_id, round((l1+l2)/2, 3))
                    #save_pred_gt(gt, pred, file_name)

print(' valide # queries: ', cpt_valid_queries)
json.dump(vq3d_queries, open(output_filename, 'w'))
print("min depth:",min(depth_list))
print("max depth:",max(depth_list))
print("mean depth:",np.mean(depth_list))
# # Guarda los centros annot
#pcd = o3d.geometry.PointCloud()
#pcd.points = o3d.utility.Vector3dVector(annot_centers_list)
#o3d.io.write_point_cloud("/media/SSD0/mcescobar/episodic-memory/colmap/centers/centers_annot_w_rotation.ply", pcd)

# # Guarda los centros predichos
#pcd = o3d.geometry.PointCloud()
#pcd.points = o3d.utility.Vector3dVector(pred_centers_list)
#o3d.io.write_point_cloud("/media/SSD0/mcescobar/episodic-memory/colmap/centers/pred_centers_constrain.ply", pcd)
#breakpoint()
#visualize_points(annot_centers_list,pred_centers_list, "/media/SSD0/mcescobar/episodic-memory/colmap/centers/lines")