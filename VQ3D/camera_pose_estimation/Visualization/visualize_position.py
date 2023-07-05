import numpy as np
import open3d as o3d

# Load the 3D scan data
mesh = o3d.io.read_triangle_mesh("3d_scan.obj")

# Create a camera intrinsic matrix
intrinsics = o3d.camera.PinholeCameraIntrinsic(width=640, height=480, fx=500, fy=500, cx=320, cy=240)

# Create a camera extrinsic matrix
R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
t = np.array([0, 0, 0])
extrinsics = np.hstack((R, t.reshape(3, 1)))

# Create a PinholeCamera object
camera = o3d.camera.PinholeCamera(intrinsics, extrinsics)

# Visualize the point of view from the camera
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(mesh)
vis.get_render_option().load_from_json('{"point_size": 2.0}')
vis.get_view_control().convert_from_pinhole_camera_parameters(camera.intrinsic, camera.extrinsic)
vis.run()
vis.destroy_window()