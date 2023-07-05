import json
import sys
import os
sys.path.append('API/')
from get_query_3d_ground_truth import VisualQuery3DGroundTruth

path = "/media/SSD0/mcescobar/episodic-memory/VQ3D/data/vq3d_results/siam_rcnn_residual_kys_val.json"
save_path = "/media/SSD0/mcescobar/episodic-memory/VQ3D/data/submissions"
name_exp = "test_baseline"
output = {"version": "1.0", 
            "challenge": "ego4d_vq3d_challenge",
            "results": {
                "videos":[]
            }
}
poses_root = "/media/SSD5/ego4d/dataset/3d/v1/clips_camera_poses_5fps"
helper = VisualQuery3DGroundTruth()
pred = json.load(open(path, "r"))
for video in pred["videos"]:
    video_uid= video["video_uid"]
    video_dict = {"video_uid":video_uid, "clips":[]}
    for clip in video["clips"]:
        clip_uid = clip["clip_uid"]
        root_pos_clip = os.path.join(poses_root, clip_uid, "egovideo")
        poses = helper.load_pose(root_pos_clip)
        clip_dict = {"clip_uid": clip_uid, "predictions":[]} # TODO: what to do when the annotation is {}
        for ai, annot in enumerate(clip['annotations']):
            if not annot:
                clip_dict["predictions"].append({})
                continue # TODO: Ask Vince #TODO: necesitamos el test map
            query_sets = {}
            for qset_id, qset in annot['query_sets'].items():
                dict_query = {}
                if not 'pred_3d_vec_world' in qset:
                    dict_query["pred_3d_vec_world"] = None  
                else:
                    dict_query["pred_3d_vec_world"] = qset["pred_3d_vec_world"]
                if 'pred_3d_vec' in qset:
                    dict_query["pred_3d_vec_query_frame"] = qset["pred_3d_vec"]
                else:
                    dict_query["pred_3d_vec_query_frame"] = None
                if poses is None:
                    dict_query["query_frame_pose"] = None
                else:
                    T, valid_pose = poses
                query_sets[int(qset_id)] = dict_query
            clip_dict["predictions"].append({"query_sets":query_sets})
        video_dict["clips"].append(clip_dict)
    output["results"]["videos"].append(video_dict)
json.dump(output,open(os.path.join(save_path, name_exp+".json"),"w"), indent = 2)

            
                
                
                
                
                
