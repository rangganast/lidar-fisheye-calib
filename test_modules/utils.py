import os
import yaml
from numpy.linalg import inv
import numpy as np
import torch
import mathutils
import matplotlib.pyplot as plt


def load_calib_gt(rootdir):
    # camera extrinsic and intrinsic parameter transformation
    cam_to_pose_path = os.path.join(rootdir, "calibration", "calib_cam_to_pose.txt")
    cam_to_velo_path = os.path.join(rootdir, "calibration", "calib_cam_to_velo.txt")

    data_cam_to_pose = {}
    data_cam_to_velo = {}

    with open(cam_to_pose_path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':',1)

            try:
                data_cam_to_pose[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    with open(cam_to_velo_path, 'r') as f:
        for line in f.readlines():
            try:
                data_cam_to_velo = np.array([float(x) for x in line.split()])
            except ValueError:
                pass

    T_cam0_to_pose = np.reshape(data_cam_to_pose['image_00'], (3, 4))
    T_cam1_to_pose = np.reshape(data_cam_to_pose['image_01'], (3, 4))
    T_cam2_to_pose = np.reshape(data_cam_to_pose['image_02'], (3, 4))
    T_cam3_to_pose = np.reshape(data_cam_to_pose['image_03'], (3, 4))
    T_cam0_to_velo = np.reshape(data_cam_to_velo, (3, 4))

    last_row = np.array([0, 0, 0, 1])

    T_cam0_to_pose = np.vstack((T_cam0_to_pose, last_row))
    T_cam1_to_pose = np.vstack((T_cam1_to_pose, last_row))
    T_cam2_to_pose = np.vstack((T_cam2_to_pose, last_row))
    T_cam3_to_pose = np.vstack((T_cam3_to_pose, last_row))
    T_cam0_to_velo = np.vstack((T_cam0_to_velo, last_row))

    T_velo_to_cam0 = inv(T_cam0_to_velo)
    T_velo_to_cam1 = inv(T_cam1_to_pose).dot(T_cam0_to_pose).dot(inv(T_cam0_to_velo))
    T_velo_to_cam2 = inv(T_cam2_to_pose).dot(T_cam0_to_pose).dot(inv(T_cam0_to_velo))
    T_velo_to_cam3 = inv(T_cam3_to_pose).dot(T_cam0_to_pose).dot(inv(T_cam0_to_velo))

    return T_velo_to_cam0, T_velo_to_cam1, T_velo_to_cam2, T_velo_to_cam3


def generate_misalignment(max_rot = 30, max_trans = 0.5):
    rot_z = np.random.uniform(-max_rot, max_rot) * (3.141592 / 180.0)
    rot_y = np.random.uniform(-max_rot, max_rot) * (3.141592 / 180.0)
    rot_x = np.random.uniform(-max_rot, max_rot) * (3.141592 / 180.0)
    trans_x = np.random.uniform(-max_trans, max_trans)
    trans_y = np.random.uniform(-max_trans, max_trans)
    trans_z = np.random.uniform(-max_trans, max_trans)

    R_perturb = mathutils.Euler((rot_x, rot_y, rot_z)).to_matrix()
    t_perturb = np.array([[trans_x, trans_y, trans_z]]).reshape(3, 1)
        
    return np.array(R_perturb), t_perturb

def load_fisheye_intrinsic(rootdir, camera_id):
    intrinsic_path = os.path.join(rootdir, 
                                    "calibration", 
                                    "image_%s.yaml"%(camera_id))
    
    with open(intrinsic_path, 'r') as file:
        intrinsic = yaml.safe_load(file)

    # 2D image dimension
    H = intrinsic['image_height']
    W = intrinsic['image_width']
    img_size = (H, W)
    
    # mirror parameters
    xi = intrinsic['mirror_parameters']['xi']

    # distortion parameters
    k1 = intrinsic['distortion_parameters']['k1']
    k2 = intrinsic['distortion_parameters']['k2']
    p1 = intrinsic['distortion_parameters']['p1']
    p2 = intrinsic['distortion_parameters']['p2']

    # projection parameters
    gamma1 = intrinsic['projection_parameters']['gamma1']
    gamma2 = intrinsic['projection_parameters']['gamma2']
    u0 = intrinsic['projection_parameters']['u0']
    v0 = intrinsic['projection_parameters']['v0']
    
    # intrinsic matrix (K) formulation
    K_int = np.array([[gamma1, 0.0, u0],[0.0, gamma2, v0], [0.0, 0.0, 1.0]])

    # mirror & distortion parameters dictionary
    distort_params = {'xi': xi, 'k1': k1, 'k2': k2, 'p1': p1, 'p2': p2}

    return img_size, K_int, distort_params

def rot2qua_torch(matrix):
    """
    Convert a rotation matrix to quaternion.
    Args:
        matrix (torch.Tensor): [4x4] transformation matrix or [3,3] rotation matrix.

    Returns:
        torch.Tensor: shape [4], normalized quaternion
    """
    if matrix.shape == (4, 4):
        R = matrix[:-1, :-1]
    elif matrix.shape == (3, 3):
        R = matrix
    else:
        raise TypeError("Not a valid rotation matrix")
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    q = torch.zeros(4, device=matrix.device)
    if tr > 0.:
        S = (tr+1.0).sqrt() * 2
        q[0] = 0.25 * S
        q[1] = (R[2, 1] - R[1, 2]) / S
        q[2] = (R[0, 2] - R[2, 0]) / S
        q[3] = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = (1.0 + R[0, 0] - R[1, 1] - R[2, 2]).sqrt() * 2
        q[0] = (R[2, 1] - R[1, 2]) / S
        q[1] = 0.25 * S
        q[2] = (R[0, 1] + R[1, 0]) / S
        q[3] = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = (1.0 + R[1, 1] - R[0, 0] - R[2, 2]).sqrt() * 2
        q[0] = (R[0, 2] - R[2, 0]) / S
        q[1] = (R[0, 1] + R[1, 0]) / S
        q[2] = 0.25 * S
        q[3] = (R[1, 2] + R[2, 1]) / S
    else:
        S = (1.0 + R[2, 2] - R[0, 0] - R[1, 1]).sqrt() * 2
        q[0] = (R[1, 0] - R[0, 1]) / S
        q[1] = (R[0, 2] + R[2, 0]) / S
        q[2] = (R[1, 2] + R[2, 1]) / S
        q[3] = 0.25 * S
    return q / q.norm()

def qua2rot_torch(q):
    """
    Convert a quaternion to a rotation matrix
    Args:
        q (torch.Tensor): shape [4], input quaternion

    Returns:
        torch.Tensor: [3x3] homogeneous rotation matrix
    """
    assert q.shape == torch.Size([4]), "Not a valid quaternion"
    if q.norm() != 1.:
        q = q / q.norm()
    mat = torch.zeros((3, 3), device=q.device)
    mat[0, 0] = 1 - 2*q[2]**2 - 2*q[3]**2
    mat[0, 1] = 2*q[1]*q[2] - 2*q[3]*q[0]
    mat[0, 2] = 2*q[1]*q[3] + 2*q[2]*q[0]
    mat[1, 0] = 2*q[1]*q[2] + 2*q[3]*q[0]
    mat[1, 1] = 1 - 2*q[1]**2 - 2*q[3]**2
    mat[1, 2] = 2*q[2]*q[3] - 2*q[1]*q[0]
    mat[2, 0] = 2*q[1]*q[3] - 2*q[2]*q[0]
    mat[2, 1] = 2*q[2]*q[3] + 2*q[1]*q[0]
    mat[2, 2] = 1 - 2*q[1]**2 - 2*q[2]**2
    return mat

class depth_rgb_proj:
    def __init__(self, image_size, K_int, distort_params):
        self.H, self.W = image_size
        
        # intrinsic parameters
        self.K_int = K_int 
        self.K_int = torch.tensor(self.K_int).type(torch.float32)

        # mirror and distortion parameters
        self.xi = distort_params['xi']
        self.k1 = distort_params['k1']
        self.k2 = distort_params['k2']
        self.p1 = distort_params['p1']
        self.p2 = distort_params['p2']
        
    def __call__(self, point_cloud, T_ext, rgb_img, device, image_path='test_calib.png'):
        rgb_img = rgb_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        
        point_cloud = point_cloud.squeeze().to(device)
        T_ext = T_ext.squeeze(0).to(device)
        self.K_int = self.K_int.to(device)
     
        # extrinsic transformation from velodyne reference frame to fisheye camera reference frame
        n_points = point_cloud.shape[0]
        pcd_fisheye = torch.matmul(T_ext, torch.hstack((point_cloud, torch.ones((n_points, 1)).to(device))).T).T  # (P_velo -> P_fisheye)
        pcd_fisheye = pcd_fisheye[:,:3]
        z_axis = pcd_fisheye[:,2]  # Extract the z_axis to determine 3D points that located in front of the camera.

        # Project to unit sphere (every vector magnitude is 1)
        pcd_sphere = pcd_fisheye / torch.linalg.norm(pcd_fisheye, dim=1, keepdim=True)

        # Reference frame center moved to Cp = xi => (X, Y, Z) --> (X, Y, Z + Xi)

        pcd_sphere[:,2] += self.xi

        # Project into normalized plane => (X/(Z + Xi), Y/(Z + Xi), 1)
        norm_plane = pcd_sphere / pcd_sphere[:, 2].unsqueeze(1)

        # distortion ((u',v') = radial(u,v) + tangential(u,v))
        distorted_plane = self.distortion(norm_plane)

        # final projection using generalized camera projection (intrinsic matrix K)
        distorted_pixels = torch.cat((distorted_plane, norm_plane[:, 2].unsqueeze(1)), dim=1)
        pixel_proj = torch.matmul(self.K_int, distorted_pixels.T).T

        # Convert pixel_proj to integer indices for projection
        u = pixel_proj[:, 0].to(torch.int32)
        v = pixel_proj[:, 1].to(torch.int32)

        # Compute depth for each point in the point cloud
        depth = torch.norm(point_cloud, dim=1)

        condition = (0<=u)*(u<self.W)*(0<=v)*(v<self.H)*(depth>0)*(z_axis>=0)
        # print(np.min(z_axis))

        u_proj = u[condition]
        v_proj = v[condition]
        d_proj = depth[condition]
        
        max_depth = torch.max(d_proj).to(device)
        d_proj = torch.clamp((d_proj / max_depth) * 255, 0, 255).to(device)

        # # image array generation
        # image_tensor = torch.zeros(self.H, self.W, dtype=torch.float32).to(device)
        # image_tensor[v_proj,u_proj] = d_proj

        # # expand dimension to (1400, 1400, 3)
        # image_tensor = torch.unsqueeze(image_tensor, 0)
        # image_tensor = image_tensor.expand(3, -1, -1)
        
        # save_image(image_tensor, "test_image.png")
        
        u_proj = u_proj.detach().cpu().numpy()
        v_proj = v_proj.detach().cpu().numpy()
        d_proj = d_proj.detach().cpu().numpy()
        
        fig = plt.figure(figsize=(14,14),dpi=100,frameon=False)
        # fig.set_size(self.W, self.H)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        ax.imshow(rgb_img)
        ax.scatter([u_proj],[v_proj],c=[d_proj],cmap='rainbow_r',alpha=0.3,s=10)
        # fig.show()
        fig.savefig(image_path)
        
    def distortion(self, norm_plane):
        # Compute r = sqrt(x^2 + y^2) for each pixel
        r = torch.sqrt(norm_plane[:, 0]**2 + norm_plane[:, 1]**2)

        # Radial distortion
        D = 1 + self.k1 * (r**2) + self.k2 * (r**4)

        # Tangential distortion
        dx = 2 * self.p1 * norm_plane[:, 0] * norm_plane[:, 1] + self.p2 * ((r**2) + 2 * (norm_plane[:, 0]**2))
        dy = self.p1 * ((r**2) + 2 * (norm_plane[:, 1]**2)) + 2 * self.p2 * norm_plane[:, 0] * norm_plane[:, 1]

        # Total distortion
        x_distorted = D * norm_plane[:, 0] + dx
        y_distorted = D * norm_plane[:, 1] + dy

        # Combine into a distorted plane tensor
        distorted_plane = torch.stack([x_distorted, y_distorted], dim=1)
        return distorted_plane
        
class depth_image_projection_fisheye_torch:
    def __init__(self, image_size, K_int, distort_params):
        self.H, self.W = image_size
        
        # intrinsic parameters
        self.K_int = K_int 

        # mirror and distortion parameters
        self.xi, self.k1, self.k2, self.p1, self.p2 = distort_params
        
    def __call__(self, point_cloud, T_ext, device):
        point_cloud = point_cloud.to(device)
        T_ext = T_ext.to(device)
        self.K_int = self.K_int.to(device)
     
        # extrinsic transformation from velodyne reference frame to fisheye camera reference frame
        n_points = point_cloud.shape[0]
        pcd_fisheye = torch.matmul(T_ext, torch.hstack((point_cloud, torch.ones((n_points, 1)).to(device))).T).T  # (P_velo -> P_fisheye)
        pcd_fisheye = pcd_fisheye[:,:3]
        z_axis = pcd_fisheye[:,2]  # Extract the z_axis to determine 3D points that located in front of the camera.

        # Project to unit sphere (every vector magnitude is 1)
        pcd_sphere = pcd_fisheye / torch.linalg.norm(pcd_fisheye, dim=1, keepdim=True)

        # Reference frame center moved to Cp = xi => (X, Y, Z) --> (X, Y, Z + Xi)
        pcd_sphere[:,2] += self.xi

        # Project into normalized plane => (X/(Z + Xi), Y/(Z + Xi), 1)
        norm_plane = pcd_sphere / pcd_sphere[:, 2].unsqueeze(1)

        # distortion ((u',v') = radial(u,v) + tangential(u,v))
        distorted_plane = self.distortion(norm_plane)

        # final projection using generalized camera projection (intrinsic matrix K)
        distorted_pixels = torch.cat((distorted_plane, norm_plane[:, 2].unsqueeze(1)), dim=1)
        pixel_proj = torch.matmul(self.K_int, distorted_pixels.T).T

        # Convert pixel_proj to integer indices for projection
        u = pixel_proj[:, 0].to(torch.int32)
        v = pixel_proj[:, 1].to(torch.int32)

        # Compute depth for each point in the point cloud
        depth = torch.norm(point_cloud, dim=1)

        condition = (0<=u)*(u<self.W)*(0<=v)*(v<self.H)*(depth>0)*(z_axis>=0)
        # print(np.min(z_axis))

        u_proj = u[condition]
        v_proj = v[condition]
        d_proj = depth[condition]

        max_depth = torch.max(d_proj).to(device)
        d_proj = torch.clamp((d_proj / max_depth) * 255, 0, 255).to(device)

        # image array generation
        image_tensor = torch.zeros(self.H, self.W, dtype=torch.float32).to(device)
        image_tensor[v_proj,u_proj] = d_proj

        # expand dimension to (1400, 1400, 3)
        image_tensor = torch.unsqueeze(image_tensor, 0)
        image_tensor = image_tensor.expand(3, -1, -1)
        
        del d_proj

        return image_tensor

    def distortion(self, norm_plane):
        # Compute r = sqrt(x^2 + y^2) for each pixel
        r = torch.sqrt(norm_plane[:, 0]**2 + norm_plane[:, 1]**2)

        # Radial distortion
        D = 1 + self.k1 * (r**2) + self.k2 * (r**4)

        # Tangential distortion
        dx = 2 * self.p1 * norm_plane[:, 0] * norm_plane[:, 1] + self.p2 * ((r**2) + 2 * (norm_plane[:, 0]**2))
        dy = self.p1 * ((r**2) + 2 * (norm_plane[:, 1]**2)) + 2 * self.p2 * norm_plane[:, 0] * norm_plane[:, 1]

        # Total distortion
        x_distorted = D * norm_plane[:, 0] + dx
        y_distorted = D * norm_plane[:, 1] + dy

        # Combine into a distorted plane tensor
        distorted_plane = torch.stack([x_distorted, y_distorted], dim=1)
        return distorted_plane