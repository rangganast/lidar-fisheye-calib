import open3d as o3d
import numpy as np
from numpy.linalg import inv
import yaml
import os
import math
import torch

class pcd_extrinsic_transform: # transform PCD into fisheye camera reference frame.
    def __init__(self, crop = True):
        self.crop = crop

    def __call__(self, point_cloud, T_ext):
        # extrinsic transformation from velodyne reference frame to fisheye camera reference frame
        n_points = point_cloud.shape[0]

        # Add homogeneous coordinate to point cloud
        ones = torch.ones((n_points, 1), device=point_cloud.device, dtype=point_cloud.dtype)
        point_cloud_h = torch.hstack((point_cloud, ones))  # (N, 4)

        # Apply extrinsic transformation
        pcd_cam = torch.matmul(T_ext, point_cloud_h.T).T  # (N, 4)
        pcd_cam = pcd_cam[:, :3]  # Extract 3D coordinates (N, 3)

        # Extract the z-axis to determine points in front of the camera
        z_axis = pcd_cam[:, 2]

        # Apply cropping if enabled
        if self.crop:
            condition = z_axis >= 0
            new_pcd = pcd_cam[condition]
        else:
            new_pcd = pcd_cam

        return new_pcd
    
class pcd_extrinsic_transform_torch: # transform PCD into fisheye camera reference frame.
    def __init__(self, crop = True):
        self.crop = crop

    def __call__(self, point_cloud, T_ext):
        # extrinsic transformation from velodyne reference frame to fisheye camera reference frame
        device_ = point_cloud.device
        n_points = point_cloud.shape[0]
        pcd_fisheye = torch.matmul(T_ext, torch.hstack((point_cloud, torch.ones(n_points, 1).to(device_))).T).T  # (P_velo -> P_fisheye)
        pcd_fisheye = pcd_fisheye[:,:3]
        z_axis = pcd_fisheye[:,2]  # Extract the z_axis to determine 3D points that located in front of the camera.

        if self.crop:
            condition = (z_axis>=0)
            new_pcd = pcd_fisheye[condition]
        else:
            new_pcd = pcd_fisheye

        # print(point_cloud.shape)
        # print(new_pcd.shape)

        return new_pcd

class depth_image_projection_fisheye:
    def __init__(self, image_size, K_int, distort_params):
        self.H, self.W = image_size
        
        # intrinsic parameters
        self.K_int = K_int 

        # mirror and distortion parameters
        self.xi, self.k1, self.k2, self.p1, self.p2 = distort_params
        
    def __call__(self, point_cloud, T_ext):
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

        # image array generation
        image_tensor = torch.zeros(self.H, self.W, dtype=torch.float32)
        image_tensor[v_proj,u_proj] = torch.from_numpy(d_proj).type(torch.float32) #(1400, 1400, )

        # expand dimension to (1400, 1400, 3)
        image_tensor = torch.unsqueeze(image_tensor, 0)
        image_tensor = image_tensor.expand(3, -1, -1)

        return image_tensor

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

# Point Cloud Transformation / Pre-processing -> remove outlier, voxel downsample, norms normalization
class point_cloud_transform:
    def __init__(self, voxel_size = 0.3, remove_outlier = None, concat = None, **kwargs):
        self.voxel_size = voxel_size
        self.remove_outlier = remove_outlier
        self.n_neighbor = kwargs.get('n_neighbor', 20)
        self.std_ratio = kwargs.get('std_ratio', 2.0)
        self.max_nn = kwargs.get('max_nn', 30)
        self.concat = concat

    def __call__(self, x):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(x)

        if self.remove_outlier is not None:
            if self.remove_outlier == 'radius':
                _, ind = pcd.remove_radius_outlier(nb_points=self.n_neighbor, radius=self.voxel_size)
                pcd.select_by_index(ind)
            elif self.remove_outlier == 'statistical':
                _, ind = pcd.remove_statistical_outlier(nb_points=self.n_neighbor, std_ratio=self.std_ratio)
                pcd.select_by_index(ind)
            else:
                pass

        pcd = pcd.voxel_down_sample(self.voxel_size)
        
        pcd_arr = np.array(pcd.points, dtype=np.float32)

        if self.concat is None:
            return pcd_arr
        else:
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size*3, max_nn=30))
            pcd.normalize_normals()
            pcd_norm = np.array(pcd.normals,dtype=np.float32)

            if self.concat == 'xyz':
                return np.hstack([pcd_arr,pcd_norm])  # (N,3), (N,3) -> (N,6)
            elif self.concat == 'zero-mean':  # 'zero-mean'
                center = np.mean(pcd_arr,axis=0,keepdims=True)  # (3,)
                pcd_zero = pcd_arr - center
                pcd_norm *= np.where(np.sum(pcd_zero*pcd_norm,axis=1,keepdims=True)<0,-1,1)
                return np.hstack([pcd_arr,pcd_norm]) # (N,3),(N,3) -> (N,6)
            else:
                raise RuntimeError('Unknown concat mode: %s'%self.concat)

def load_pcd(path):
    pcd = np.fromfile(path, dtype=np.float32)
    return pcd.reshape((-1, 4))

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

def load_cam_intrinsic(rootdir = "./datasets/KITTI-360/", camera_id = '01'):
    calib_params = {}
    with open(os.path.join(rootdir, 'calibration', 'perspective.txt'), 'r') as file:
        for line in file.readlines():
            try:
                key, value = line.split(':', 1)
            except ValueError:
                key, value = line.split(' ', 1)
            try:
                calib_params[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    im_size = (int(calib_params['S_rect_'+ camera_id][0]), 
                    int(calib_params['S_rect_'+ camera_id][1]))
    
    K_int = (np.resize(calib_params['P_rect_' + camera_id], (3,4)))[:3,:3]

    return im_size, K_int

def rot2qua(matrix):
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
    q = np.zeros(4)
    if tr > 0.:
        S = math.sqrt(tr+1.0) * 2
        q[0] = 0.25 * S
        q[1] = (R[2, 1] - R[1, 2]) / S
        q[2] = (R[0, 2] - R[2, 0]) / S
        q[3] = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q[0] = (R[2, 1] - R[1, 2]) / S
        q[1] = 0.25 * S
        q[2] = (R[0, 1] + R[1, 0]) / S
        q[3] = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q[0] = (R[0, 2] - R[2, 0]) / S
        q[1] = (R[0, 1] + R[1, 0]) / S
        q[2] = 0.25 * S
        q[3] = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q[0] = (R[1, 0] - R[0, 1]) / S
        q[1] = (R[0, 2] + R[2, 0]) / S
        q[2] = (R[1, 2] + R[2, 1]) / S
        q[3] = 0.25 * S
    return q / np.linalg.norm(q)

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

def qua2rot(q):
    """
    Convert a quaternion to a rotation matrix
    Args:
        q (torch.Tensor): shape [4], input quaternion

    Returns:
        torch.Tensor: [3x3] homogeneous rotation matrix
    """
    assert q.shape[0] == 4, "Not a valid quaternion"
    if np.linalg.norm(q) != 1.:
        q = q / np.linalg.norm(q)
    mat = np.zeros((3, 3))
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

# call test
if __name__ == "__main__":
    # q = np.array([0.23757144, 0.5996278,  0.28553449, 0.70885568])
    # R = qua2rot(q)
    # print(R)
    
    # q = rot2qua(R)
    # print(q)
    R = np.array([[0.99922916, 0.03655203, 0.0143193],
                  [0.01458545, -0.0070234, -0.99986896],
                  [-0.03644667, 0.99930707, -0.00755111]])
    
    q = rot2qua(R)
    print(q)

    R = qua2rot(q)
    print(R)
    