import os
import numpy as np
import math
import matplotlib.pyplot as plt
from torchvision.io import read_image, ImageReadMode

from test_modules.utils import load_calib_gt, load_fisheye_intrinsic, generate_misalignment, qua2rot_torch

class pcd_rgb_proj:
    def __init__(self, image_size, K_int, distort_params):
        self.H, self.W = image_size
        
        # intrinsic parameters
        self.K_int = K_int

        # mirror and distortion parameters
        self.xi = distort_params['xi']
        self.k1 = distort_params['k1']
        self.k2 = distort_params['k2']
        self.p1 = distort_params['p1']
        self.p2 = distort_params['p2']
        
    def __call__(self, point_cloud, T_ext, rgb_img, image_path='test_calib.png'):
        rgb_img = rgb_img.permute(1, 2, 0)
        
        # extrinsic transformation from velodyne reference frame to fisheye camera reference frame
        n_points = point_cloud.shape[0]
        pcd_fisheye = np.matmul(T_ext, np.hstack((point_cloud, np.ones((n_points, 1)))).T).T  # (P_velo -> P_fisheye)
        pcd_fisheye = pcd_fisheye[:,:3]
        z_axis = pcd_fisheye[:,2]  # Extract the z_axis to determine 3D points that located in front of the camera.

        # Project to unit sphere (every vector magnitude is 1)
        pcd_sphere = np.array([x/np.linalg.norm(x) for x in pcd_fisheye])

        # Reference frame center moved to Cp = xi => (X, Y, Z) --> (X, Y, Z + Xi)
        pcd_sphere[:,2] += self.xi

        # Project into normalized plane => (X/(Z + Xi), Y/(Z + Xi), 1)
        norm_plane = np.array([x/x[2] for x in pcd_sphere])

        # distortion ((u',v') = radial(u,v) + tangential(u,v))
        distorted_plane = np.array([self.distortion(pixel[0], pixel[1]) for pixel in norm_plane])

        # final projection using generalized camera projection (intrinsic matrix K)
        distorted_pixels = np.hstack((distorted_plane, np.expand_dims(norm_plane[:,2], axis=1)))
        pixel_proj = np.matmul(self.K_int, distorted_pixels.T).T

        u = np.asarray(pixel_proj[:, 0], dtype=np.int32)
        v = np.asarray(pixel_proj[:, 1], dtype=np.int32)

        # depth calculation of each point
        depth = np.array([np.linalg.norm(x) for x in point_cloud])

        condition = (0<=u)*(u<self.W)*(0<=v)*(v<self.H)*(depth>0)*(z_axis>=0)
        # print(np.min(z_axis))

        u_proj = u[condition]
        v_proj = v[condition]
        d_proj = depth[condition]

        max_depth = np.max(d_proj)
        d_proj = np.array([np.interp(d, [0, max_depth], [0, 255]) for d in d_proj]) # convert depth values to [0, 255]
        
        fig = plt.figure(figsize=(14,14),dpi=100,frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        ax.imshow(rgb_img)
        ax.scatter([u_proj],[v_proj],c=[d_proj],cmap='rainbow_r',alpha=0.3,s=10)
        fig.savefig(image_path)
        plt.close(fig)
        
    def distortion(self, x, y):
        r = math.sqrt(x**2 + y**2)

        # radial distortion
        D = 1 + (self.k1)*(r**2) + (self.k2)*(r**4)

        # tangential distortion
        dx = 2*self.p1*x*y + self.p2*((r**2) + 2*(x**2))
        dy = self.p1*((r**2) + 2*(y**2)) + 2*self.p2*x*y

        # total distortion
        x_distorted = D*x + dx
        y_distorted = D*y + dy

        return np.array([x_distorted, y_distorted])

class pcd_depth_proj:
    def __init__(self, image_size, K_int, distort_params):
        self.H, self.W = image_size
        
        # intrinsic parameters
        self.K_int = K_int

        # mirror and distortion parameters
        self.xi = distort_params['xi']
        self.k1 = distort_params['k1']
        self.k2 = distort_params['k2']
        self.p1 = distort_params['p1']
        self.p2 = distort_params['p2']
        
    def __call__(self, point_cloud, T_ext, image_path='depth_output.png'):
        # Extrinsic transformation
        n_points = point_cloud.shape[0]
        homog_points = np.hstack((point_cloud, np.ones((n_points, 1))))
        pcd_fisheye = (T_ext @ homog_points.T).T[:, :3]
        z_axis = pcd_fisheye[:, 2]

        # Project to unit sphere
        norms = np.linalg.norm(pcd_fisheye, axis=1, keepdims=True)
        pcd_sphere = pcd_fisheye / norms

        # Reference frame adjustment
        pcd_sphere[:, 2] += self.xi

        # Normalized plane projection
        norm_plane = pcd_sphere / pcd_sphere[:, [2]]  # Keep dimensions for broadcasting

        # Apply your existing distortion function
        distorted_plane = np.array([self.distortion(p[0], p[1]) for p in norm_plane[:, :2]])

        # Final projection
        distorted_pixels = np.hstack((distorted_plane, norm_plane[:, [2]]))
        pixel_proj = (self.K_int @ distorted_pixels.T).T

        # Pixel coordinates and depth calculation
        u = np.round(pixel_proj[:, 0]).astype(int)
        v = np.round(pixel_proj[:, 1]).astype(int)
        depth = np.linalg.norm(point_cloud, axis=1)

        # Filter valid points
        valid = (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H) & (depth > 0) & (z_axis >= 0)
        u_proj, v_proj, d_proj = u[valid], v[valid], depth[valid]

        # Normalize depth values
        max_depth = np.max(d_proj)
        d_proj_normalized = (d_proj / max_depth * 255).astype(np.uint8)

        # Create depth image
        depth_image = np.zeros((self.H, self.W), dtype=np.uint8)
        depth_image[v_proj, u_proj] = d_proj_normalized

        # Save using matplotlib
        plt.figure(figsize=(14, 14))
        plt.imshow(depth_image*255, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
    def distortion(self, x, y):
        r = math.sqrt(x**2 + y**2)

        # radial distortion
        D = 1 + (self.k1)*(r**2) + (self.k2)*(r**4)

        # tangential distortion
        dx = 2*self.p1*x*y + self.p2*((r**2) + 2*(x**2))
        dy = self.p1*((r**2) + 2*(y**2)) + 2*self.p2*x*y

        # total distortion
        x_distorted = D*x + dx
        y_distorted = D*y + dy

        return np.array([x_distorted, y_distorted])

def load_img(path):
    image = read_image(path, ImageReadMode.RGB)

    return image

def load_pcd(path):
    pcd = np.fromfile(path, dtype=np.float32)
    return pcd.reshape((-1, 4))

def generate_images(pcds, rgb_imgs, max_rot, max_trans):
    img_size, K, distort_params = load_fisheye_intrinsic(rootdir = "/home/rangganast/rangganast/dataset/KITTI-360", camera_id = "02")
    
    proj_pcd_to_rgb = pcd_rgb_proj(image_size=img_size,
                            K_int = K,
                            distort_params=distort_params)
    
    proj_pcd_to_depth = pcd_depth_proj(image_size=img_size,
                            K_int = K,
                            distort_params=distort_params)
    
    T_velo_to_cam_gt = load_calib_gt(rootdir="/home/rangganast/rangganast/dataset/KITTI-360")[2]
    
    delta_R_gt, delta_t_gt = generate_misalignment(max_rot, max_trans)
    delta_T = np.hstack((delta_R_gt, delta_t_gt))
    delta_T = np.vstack((delta_T, np.array([0.0, 0.0, 0.0, 1.0])))
    
    T_mis = np.matmul(delta_T, T_velo_to_cam_gt)
    
    for i in range(len(pcds)):
        rgb_img = load_img(rgb_imgs[i])
        pcd = load_pcd(pcds[i])
        pcd = pcd[:, :3]
        
        x = proj_pcd_to_depth(pcd, T_velo_to_cam_gt, f"figure/ground_truth_proj_depth_{i}.png")
        x = proj_pcd_to_depth(pcd, T_mis, f"figure/misaligned_proj_depth_{i}.png")
        
        x = proj_pcd_to_rgb(pcd, T_velo_to_cam_gt, rgb_img, f"figure/ground_truth_proj_rgb_{i}.png")
        x = proj_pcd_to_rgb(pcd, T_mis, rgb_img, f"figure/misaligned_proj_rgb_{i}.png")

if __name__ == "__main__":
    rgb_imgs = [
        "/home/rangganast/rangganast/dataset/KITTI-360/data_2d_raw/2013_05_28_drive_0000_sync/image_02/data_rgb/0000000000.png",
        # "/home/rangganast/rangganast/dataset/KITTI-360/data_2d_raw/2013_05_28_drive_0003_sync/image_02/data_rgb/0000000000.png",
        # "/home/rangganast/rangganast/dataset/KITTI-360/data_2d_raw/2013_05_28_drive_0004_sync/image_02/data_rgb/0000000000.png",
        # "/home/rangganast/rangganast/dataset/KITTI-360/data_2d_raw/2013_05_28_drive_0005_sync/image_02/data_rgb/0000000000.png",
    ]
    
    pcds = [
        "/home/rangganast/rangganast/dataset/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/0000000000.bin",
        # "/home/rangganast/rangganast/dataset/KITTI-360/data_3d_raw/2013_05_28_drive_0003_sync/velodyne_points/data/0000000000.bin",
        # "/home/rangganast/rangganast/dataset/KITTI-360/data_3d_raw/2013_05_28_drive_0004_sync/velodyne_points/data/0000000000.bin",
        # "/home/rangganast/rangganast/dataset/KITTI-360/data_3d_raw/2013_05_28_drive_0005_sync/velodyne_points/data/0000000000.bin",
    ]
    
    max_rot = 10.0 # deg
    max_trans = 1.0 # meter
    
    generate_images(pcds, rgb_imgs, max_rot, max_trans)