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
