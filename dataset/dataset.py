import os
import glob
import numpy as np
import open3d as o3d
import mathutils
import random
import warnings
import paramiko
import fnmatch
import struct
import time

import torch
import torch.utils
import torch.utils.data
from torch.utils.data.dataset import Dataset 
from torchvision.transforms import transforms as T
import torchvision.transforms.functional as TTF
from torchvision.io import read_image, ImageReadMode

warnings.filterwarnings("ignore")

from .helpers import (pcd_extrinsic_transform,
                     depth_image_projection_fisheye_torch,
                     load_calib_gt, 
                     load_fisheye_intrinsic,
                     rot2qua_torch) 

class KITTI360_Fisheye_Dataset(Dataset):
    def __init__(self,
                 rootdir="./KITTI-360", 
                 sequences = [0], 
                 camera_id = "02",
                 frame_step = 1,
                 n_scans = None,
                 voxel_size = None,
                 max_trans = [1.5, 1.0, 0.5, 0.2, 0.1],
                 max_rot = [20, 10, 5, 2, 1],
                 rgb_transform = None,
                 depth_transform = None,
                 split = None,
                 return_pcd = False,
                 device = 'cuda'):
        super(KITTI360_Fisheye_Dataset, self).__init__()

        self.rootdir = rootdir
        self.sequences = sequences
        self.camera_id = camera_id
        self.scans = []
        self.voxel_size = voxel_size
        self.max_trans = max_trans
        self.max_rot = max_rot
        self.rgb_transform = T.ToTensor() if rgb_transform is None else rgb_transform
        self.depth_transform = depth_transform
        self.split = split
        self.return_pcd = return_pcd
        self.device = device        

        if self.camera_id == "00":
            self.T_velo_to_cam_gt = load_calib_gt(self.rootdir)[0]
        elif self.camera_id == "01":
            self.T_velo_to_cam_gt = load_calib_gt(self.rootdir)[1]
        elif self.camera_id == "02":
            self.T_velo_to_cam_gt = load_calib_gt(self.rootdir)[2]
        else:
            self.T_velo_to_cam_gt = load_calib_gt(self.rootdir)[3]
            
        self.T_velo_to_cam_gt = torch.from_numpy(self.T_velo_to_cam_gt).type(torch.float32)

        self.im_size, self.K_int, distort_params = load_fisheye_intrinsic(self.rootdir, self.camera_id)
        self.K_int = torch.from_numpy(self.K_int).type(torch.float32)
        
        self.distort_params = (distort_params['xi'], 
                                distort_params['k1'],
                                distort_params['k2'],
                                distort_params['p1'],
                                distort_params['p2'])
        self.depth_img_gen = depth_image_projection_fisheye_torch(image_size=self.im_size,
                                                            K_int=self.K_int,
                                                            distort_params=self.distort_params)

        
        for i in range(len(max_trans)):
            for sequence in sequences:
                img_paths = []
                pcl_paths = []
                
                img_path = os.path.join(
                    self.rootdir, 
                    "data_2d_raw", 
                    "2013_05_28_drive_%04d_sync"%(sequence), 
                    "image_%s/data_rgb"%(camera_id),
                )

                pcl_path = os.path.join(
                    self.rootdir, 
                    "data_3d_raw", 
                    "2013_05_28_drive_%04d_sync"%(sequence), 
                    "velodyne_points/data"
                )
                
                for filename in os.listdir(img_path):
                    if fnmatch.fnmatch(filename, '*.png'):
                        img_paths.append(os.path.join(img_path, filename))
                        
                for filename in os.listdir(pcl_path):
                    if fnmatch.fnmatch(filename, '*.bin'):
                        pcl_paths.append(os.path.join(pcl_path, filename))
                        
                img_paths = sorted(img_paths)
                pcl_paths = sorted(pcl_paths)

                for img, pcl in zip(img_paths, pcl_paths):
                    self.scans.append({"img_path": img,
                                        "pcl_path": pcl, 
                                        "sequence": sequence,
                                        "max_trans": max_trans[i], 
                                        "max_rot": max_rot[i]})

        scan_len = len(self.scans)
        scan_idx = list(range(0, scan_len, frame_step))
        self.scans = [self.scans[i] for i in scan_idx]

        # limit the data length
        if n_scans is not None:
            self.scans = self.scans[:n_scans]

        self.rotate_pcd = pcd_extrinsic_transform(crop=False) 
        
    def __len__(self):
        return len(self.scans)
    
    def __getitem__(self, index):    
        data = {'T_gt': torch.tensor(self.T_velo_to_cam_gt)}
        scan = self.scans[index]

        img_path = scan["img_path"]
        pcl_path = scan["pcl_path"]
        sequence = scan["sequence"]
        max_trans = scan["max_trans"]
        max_rot = scan["max_rot"]

        # filename and frame_id
        filename = os.path.basename(img_path)
        frame_id = os.path.splitext(filename)[0]

        # transform 2d fisheye image
        img = self.load_img(img_path)
        img = self.custom_rgb_transform(img)
        img = self.rgb_transform(img)

        # load and preprocess point cloud data (outlier removal & voxel downsampling)
        pcd = self.load_pcd(pcl_path)[:,:3]

        # generate misalignment in extrinsic parameters (labels)
        while True:
            delta_R_gt, delta_t_gt = self.generate_misalignment(max_rot, max_trans)
            delta_q_gt = rot2qua_torch(delta_R_gt)
            
            # Form transformation matrix delta_T
            delta_T = torch.hstack((delta_R_gt, delta_t_gt.unsqueeze(1)))
            delta_T = torch.vstack((delta_T, torch.tensor([0.0, 0.0, 0.0, 1.0])))

            # Apply the misalignment transformation
            T_mis = torch.matmul(delta_T, self.T_velo_to_cam_gt)

            # generate 2d depth image from point cloud
            depth_img_error = self.depth_img_gen(pcd, T_mis, self.device)

            # check if the depth image is totally blank or not
            if torch.count_nonzero(depth_img_error) > 0.03*torch.numel(depth_img_error) or self.camera_id == '02' or self.camera_id == '03':
                break
        
        if self.depth_transform is not None:
            depth_img_error = self.depth_transform(depth_img_error)
        else:
            depth_img_error = depth_img_error

        delta_t_gt = torch.tensor(delta_t_gt)
        delta_q_gt = torch.tensor(delta_q_gt)

        # sample for dataloader
        data["frame_id"] = frame_id
        data["img_path"] = img_path
        data["sequence"] = sequence
        data["img"] = img
        if self.return_pcd:
            data["pcd"] = pcd                # target point cloud (ground truth) if necessary
        # data['pcd_mis'] = pcd_mis           # misaligned point cloud
        # data["pcd_error"] = pcd_error
        # data["depth_img_true"] = depth_img_true     # target depth image (ground truth) if necessary
        data["depth_img_error"] = depth_img_error
        data["delta_t_gt"] = delta_t_gt             # translation error ground truth
        data["delta_q_gt"] = delta_q_gt             # rotation error ground truth
        data["T_mis"] = T_mis
        
        return data

    def load_pcd(self, path):
        pcd = np.fromfile(path, dtype=np.float32)
        pcd = torch.from_numpy(pcd).type(torch.float32)
        return pcd.reshape((-1, 4))

    def load_img(self, path):
        image = read_image(path, ImageReadMode.RGB)

        return image
    
    def custom_rgb_transform(self, rgb):
        # [0.35830889 0.40027178 0.42796574] # image_2 mean
        # [0.31525962 0.34275861 0.36938032] # image_2 std
        
        rgb = rgb / 255.
        normalization = T.Normalize(mean=[0.35830889, 0.40027178, 0.42796574],
                                        std=[0.31525962, 0.34275861, 0.36938032])
        
        if self.split == 'train':
            color_transform = T.ColorJitter(0.1, 0.1, 0.1)
            rgb = color_transform(rgb)

        rgb = normalization(rgb)
        return rgb

    def voxel_downsampling(self, points, voxel_size): # unused
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.voxel_down_sample(voxel_size)
        out_points = np.array(pcd.points, dtype=np.float32)

        return out_points

    def generate_misalignment(self, max_rot = 30, max_trans = 0.5):
        rot_z = np.random.uniform(-max_rot, max_rot) * (3.141592 / 180.0)
        rot_y = np.random.uniform(-max_rot, max_rot) * (3.141592 / 180.0)
        rot_x = np.random.uniform(-max_rot, max_rot) * (3.141592 / 180.0)
        trans_x = np.random.uniform(-max_trans, max_trans)
        trans_y = np.random.uniform(-max_trans, max_trans)
        trans_z = np.random.uniform(-max_trans, max_trans)

        R_perturb = mathutils.Euler((rot_x, rot_y, rot_z)).to_matrix()
        t_perturb = np.array([trans_x, trans_y, trans_z])
            
        return torch.tensor(R_perturb).type(torch.float32), torch.tensor(t_perturb).type(torch.float32)

    def normalize_pc(self, points): # unused
        centroid = np.mean(points, axis=0)
        points -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
        points /= furthest_distance

        return points

if __name__ == "__main__":
    import time
    
    torch.multiprocessing.set_start_method('spawn')
    
    ALL_SEQUENCE = [0,2,3,4,5,6,7,9,10]
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    RESIZE_IMG = (350, 350)
    
    rgb_transform = T.Compose([T.Resize((RESIZE_IMG[0], RESIZE_IMG[1]))])#,
                            #    T.Normalize(mean=[0.33, 0.36, 0.33], 
                            #                std=[0.30, 0.31, 0.32])])
    depth_transform = T.Compose([T.Resize((RESIZE_IMG[0], RESIZE_IMG[1]))])
    
    start = time.time()
    dataset = KITTI360_Fisheye_Dataset(rootdir="/home/wicom/360-calib/KITTI-360",
                                        sequences=[0],
                                        camera_id="02",
                                        frame_step=1,
                                        n_scans=None,
                                        voxel_size=None,
                                        max_trans=[0.5],
                                        max_rot=[10],
                                        rgb_transform=rgb_transform,
                                        depth_transform=depth_transform,
                                        device=DEVICE)
    
    dataset_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=2, shuffle=False, pin_memory=False, num_workers=0)
    
    for data in dataset_loader:
        print("ok")
        
        
