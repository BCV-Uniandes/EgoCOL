import json

json_2d = json.load(open("/media/SSD0/mcescobar/episodic-memory/VQ3D/data/v1/annotations/vq_val.json", "r"))

json_3d = json.load(open("/media/SSD0/mcescobar/episodic-memory/VQ3D/data/v1/annotations/vq3d_val.json", "r"))

clips_3d = json.load(open("/media/SSD5/ego4d/last_annotations/3d/all_clips_for_vq3d_v1.json", "r"))
videos_3d = json.load(open("/media/SSD5/ego4d/last_annotations/3d/all_videos_for_vq3d_v1.json","r"))

new_videos = []
for video in json_2d["videos"]:
    video_uid = video["video_uid"]
    if video_uid in videos_3d["val"]:
        new_videos.append(video)
total_clips = 0
for video in new_videos:
    new_clips = []
    for clip in video["clips"]:
        clip_uid = clip["clip_uid"]
        if clip_uid in clips_3d["val"]:
            new_clips.append(clip)
    total_clips += len(new_clips)
    video["clips"] = new_clips
print(total_clips)
json_2d["videos"] =  new_videos
out_file = open("/media/SSD0/mcescobar/episodic-memory/VQ3D/data/v1/annotations/my_vq2d_val.json", "w")
json.dump(json_2d,out_file, indent = 2)