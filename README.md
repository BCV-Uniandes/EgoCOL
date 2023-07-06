# EgoCOL: Egocentric Camera pose estimation for Open-world 3D object Localization
Cristhian Forigua, Maria Escobar, Jordi Pont-Tuset, Kevis-Kokitsi Maninis, Pablo Arbeláez<br/>
Center for Research and Formation in Artificial Intelligence .(CINFONIA), Universidad de los Andes, Bogotá 111711, Colombia.

[[`arXiv`](https://arxiv.org/abs/2306.16606)]

We present EgoCOL, an egocentric camera pose estimation method for open-world 3D object localization. Our method leverages sparse camera pose reconstructions in a two-fold manner, video and scan independently, to estimate the camera pose of egocentric frames in 3D renders with high recall and precision. We extensively evaluate our method on the Visual Query (VQ) 3D object localization Ego4D benchmark. EgoCOL can estimate 62%  and 59% more camera poses than the Ego4D baseline in the Ego4D Visual Queries 3D Localization challenge at CVPR 2023 in the val and test sets, respectively. 

<div align="center">
  <img src="https://github.com/BCV-Uniandes/EgoCOL/blob/main/img/Registration.drawio.png" width="100%" height="100%"/>
</div><br/>

## Installation instructions
1. Please follow the installation instructions from the [Ego4D Episodic Memory repository](https://github.com/EGO4D/episodic-memory/blob/main/VQ3D/README.md).
2. You need to install COLMAP to compute the reconstructions. Please follow [these](https://colmap.github.io/install.html) instructions to install it.
3. Finally, you need to install the Open3D library. Follow [these](http://www.open3d.org/docs/release/getting_started.html) instructions to install it.

## Data
Please follow the instructions from the Ego4D Episodic Memory repository to download the VQ3D data [here](https://github.com/EGO4D/episodic-memory/blob/main/VQ3D/README.md#data).

## Run EgoCOL
First, you need to compute the initial PnP camera poses by using the camera pose estimatio workflow proposed by Ego4D. Follow [these](https://github.com/EGO4D/episodic-memory/blob/main/VQ3D/README.md#camera-pose-estimation)
instructions to compute them. 

Once you have computed the initial camera poses you can use colmap to create the sparse reconstrutions using both the video and clip configurations:
```
$ cd colmap
$ python run_registrations.py --input_poses_dir {PATH_CLIPS_CAMERA_POSES} \
 --clips_dir {PATH_CLIPS_FRAMES} --output_dir {OUTPUT_PATH_COLMAP}
```
Similarly, run must run the registration for the scan configuration:
```
$ python run_registrations_by_scans.py --input_poses_dir {PATH_CLIPS_CAMERA_POSES} \
--clips_dir {PATH_CLIPS_FRAMES} --output_dir {OUTPUT_PATH_COLMAP_SCAN} --camera_intrinsics_filename {PATH_TO_INTRINSICS} --query_filename {PATH_TO_QUERY_ANNOT_FILE}
```
You get the folders {PATH_CLIPS_CAMERA_POSES}, {PATH_CLIPS_FRAMES}, {PATH_TO_INTRINSICS} and {PATH_TO_QUERY_ANNOT_FILE} by running the camera pose estimation worflow proposed by Ego4D. You can use the defaul value of each argument in the .py files to help you locate the right paths.

Then, you can compute the procrustes transformation between the PnP and sparse points by running the next lines. Make sure to change the paths for the "--annotations_dir", "--input_dir_colmap" and "--clips_dir" flags before you run the code.
```
$ python extract_dict_from_colmap.py
$ python extract_dict_from_colmap_by_scans.py
```
Then run the following lines:
```
$ python transform_ext.py --constrain --filter
$ python transform_ext_by_scan.py --constrain --filter
```
Make sure to change the paths for the flags. Also change the paths in lines *341* and *370* for the transform_ext.py and the lines *286*, *207* and *369* for the transform_ext_by_scan.py. The filter and constrain flags are to apply 3D constrain 

## Evaluate
### Center scan
```
$ python transform_ext.py
$ python transform_ext_by_scan.py
```
