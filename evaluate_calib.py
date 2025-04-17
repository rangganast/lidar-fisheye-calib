import os
import time
import yaml
import mathutils
import numpy as np
from numpy.linalg import inv
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms as T
from torchvision.utils import save_image

from test_modules import so3
from test_modules.quaternion_distances import quaternion_distance
from test_modules.utils import *

from dataset.dataset import KITTI360_Fisheye_Dataset
from models.model import Net  
    
def load_checkpoint(checkpoint_path, model):
    print("=> Loading checkpoint")
    print()
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    last_epoch = checkpoint["epoch"]
    last_epoch_loss = checkpoint["loss"]
    
    print("last epoch loss:", last_epoch_loss)

    return model, last_epoch, last_epoch_loss

def test_metrics(T_pred, T_gt):
    R_pred = T_pred[:,:3,:3]  # (B,3,3) rotation
    t_pred = T_pred[:,:3,3]   # (B,3) translation
    R_gt = T_gt[:,:3,:3]  # (B,3,3) rotation
    t_gt = T_gt[:,:3,3]   # (B,3) translation

    # Euclidian Distance / Absolute Distance Error Rate
    t_error = F.l1_loss(t_pred, t_gt, reduction='none')
    
    e_x = t_error[:,0].mean(dim=0)
    e_y = t_error[:,1].mean(dim=0)
    e_z = t_error[:,2].mean(dim=0)
    e_t = torch.norm(t_pred - t_gt, p=2)
    
    q_pred = rot2qua_torch(R_pred.squeeze())
    q_gt = rot2qua_torch(R_gt.squeeze())
    
    q_pred = q_pred.unsqueeze(0)
    q_gt = q_gt.unsqueeze(0)

    # Euler Angles Error Rate
    RIR = torch.bmm(torch.inverse(R_pred), R_gt)

    yaws = torch.atan2(RIR[:,1,0], RIR[:,0,0])
    pitches = torch.atan2(-RIR[:,2,0], torch.sqrt(RIR[:,2,0]*RIR[:,2,0] + RIR[:,2,2]*RIR[:,2,2]))
    rolls = torch.atan2(RIR[:,2,1], RIR[:,2,2])

    e_yaw = (torch.abs(yaws)).mean(dim=0)
    e_pitch = (torch.abs(pitches)).mean(dim=0)
    e_roll = (torch.abs(rolls)).mean(dim=0)
    e_r = quaternion_distance(q_gt, q_pred, DEVICE)

    return e_x, e_y, e_z, e_t, e_yaw, e_pitch, e_roll, e_r

def test(model, loader, batch_size):
    ex_epoch, ey_epoch, ez_epoch, et_epoch = [], [], [], []
    eyaw_epoch, eroll_epoch, epitch_epoch, er_epoch = [], [], [], []
    
    # img_size, K, distort_params = load_fisheye_intrinsic(rootdir = "/home/rangganast/rangganast/dataset/KITTI-360", camera_id = "02")
    
    # proj_img = depth_rgb_proj(image_size=img_size,
    #                         K_int = K,
    #                         distort_params=distort_params)
    
    T_velo_to_cam_gt = load_calib_gt(rootdir="/home/rangganast/rangganast/dataset/KITTI-360")[2]
    T_velo_to_cam_gt = torch.tensor(T_velo_to_cam_gt).type(torch.float32).unsqueeze(0).to(DEVICE)
    
    total_time = 0
    with torch.no_grad():
        model.eval()
        for batch_data in tqdm(loader, desc="Testing"):
            rgb_img = batch_data["img"].to(DEVICE)
            depth_img = batch_data["depth_img_error"].to(DEVICE)
            T_mis = batch_data["T_mis"].to(DEVICE)            

            # run prediction
            start = time.time()
            torch.cuda.synchronize()  # Wait for inference to complete
            delta_q_pred, delta_t_pred = model(rgb_img, depth_img)
            torch.cuda.synchronize()  # Wait for inference to complete
            end = time.time()
            
            # calculate inference time
            inf_time = end-start
            total_time += inf_time
            
            delta_R_pred = torch.stack([qua2rot_torch(q) for q in delta_q_pred])
            
            delta_T_pred = torch.zeros(batch_size, 4, 4)  # Shape: [B, 4, 4]
            delta_T_pred[:, :3, :3] = delta_R_pred
            delta_T_pred[:, :3, 3] = delta_t_pred
            delta_T_pred = delta_T_pred[:, :3, :].squeeze(0).to(DEVICE)
            delta_T_pred = torch.cat((delta_T_pred, torch.tensor([[0.0, 0.0, 0.0, 1.0]]).to(DEVICE)), dim=0).to(DEVICE)
            
            T_pred = torch.matmul(delta_T_pred.inverse(), T_mis).to(DEVICE)
            e_x, e_y, e_z, e_t, e_yaw, e_pitch, e_roll, e_r = test_metrics(T_pred, T_velo_to_cam_gt)
            
            ex_epoch.append(e_x.item())
            ey_epoch.append(e_y.item())
            ez_epoch.append(e_z.item())
            et_epoch.append(e_t.item())
            eyaw_epoch.append(e_yaw.item()*180/torch.pi)
            eroll_epoch.append(e_roll.item()*180/torch.pi)
            epitch_epoch.append(e_pitch.item()*180/torch.pi)
            er_epoch.append(e_r.item()*180/torch.pi)
            
    print("Average inference time:", total_time//len(loader))
            
    print("== X AXIS ==")
    print("x-axis error mean:", f"{100*np.asarray(ex_epoch).mean():.4f}", "cm")
    print("x-axis error median:", f"{np.median(100*np.asarray(ex_epoch)):.4f}", "cm")
    print("x-axis error std:", f"{np.std(100*np.asarray(ex_epoch)):.4f}", "cm")

    print("== Y AXIS ==")
    print("y-axis error mean:", f"{100*np.asarray(ey_epoch).mean():.4f}", "cm")
    print("y-axis error median:", f"{np.median(100*np.asarray(ey_epoch)):.4f}", "cm")
    print("y-axis error std:", f"{np.std(100*np.asarray(ey_epoch)):.4f}", "cm")

    print("== Z AXIS ==")
    print("z-axis error mean:", f"{100*np.asarray(ez_epoch).mean():.4f}", "cm")
    print("z-axis error median:", f"{np.median(100*np.asarray(ez_epoch)):.4f}", "cm")
    print("z-axis error std:", f"{np.std(100*np.asarray(ez_epoch)):.4f}", "cm")

    print("== ET ==")
    print("Translation error:", f"{100*np.asarray(et_epoch).mean():.4f}", "cm")
    print("Translation median:", f"{np.median(100*np.asarray(et_epoch)):.4f}", "cm")
    print("Translation std:", f"{np.std(100*np.asarray(et_epoch)):.4f}", "cm")

    print("== YAW ==")
    print("yaw error mean:", f"{np.asarray(eyaw_epoch).mean():.4f}", f"\N{DEGREE SIGN}")
    print("yaw error median:", f"{np.median(np.asarray(eyaw_epoch)):.4f}", f"\N{DEGREE SIGN}")
    print("yaw error std:", f"{np.std(np.asarray(eyaw_epoch)):.4f}", f"\N{DEGREE SIGN}")

    print("== PITCH ==")
    print("pitch error mean:", f"{np.asarray(epitch_epoch).mean():.4f}", f"\N{DEGREE SIGN}")
    print("pitch error median:", f"{np.median(np.asarray(epitch_epoch)):.4f}", f"\N{DEGREE SIGN}")
    print("pitch error std:", f"{np.std(np.asarray(epitch_epoch)):.4f}", f"\N{DEGREE SIGN}")
    
    print("== ROLL ==")
    print("roll error mean:", f"{np.asarray(eroll_epoch).mean():.4f}", f"\N{DEGREE SIGN}")
    print("roll error median:", f"{np.median(np.asarray(eroll_epoch)):.4f}", f"\N{DEGREE SIGN}")
    print("roll error std:", f"{np.std(np.asarray(eroll_epoch)):.4f}", f"\N{DEGREE SIGN}")
    
    print("== ER ==")
    print("Rotation error mean:", f"{np.asarray(er_epoch).mean():.4f}", f"\N{DEGREE SIGN}")
    print("Rotation error median:", f"{np.median(np.asarray(er_epoch)):.4f}", f"\N{DEGREE SIGN}")
    print("Rotation error std:", f"{np.std(np.asarray(er_epoch)):.4f}", f"\N{DEGREE SIGN}")

if __name__ == "__main__":
    print("test is starting....")
    torch.multiprocessing.set_start_method('spawn')
    torch.backends.cudnn.benchmark = True
    
    RESIZE_IMG = [350, 350]
    BATCH_SIZE = 1
    
    DEVICE = "cuda:0"
    
    VERSION = "v1"
    LOAD_CHECKPOINT_DIR = f"checkpoint_weights/run/motion_gt/{VERSION}/LiDAR_Fisheye_Calib_best.pth.tar"
    
    rgb_transform = T.Compose([T.Resize((RESIZE_IMG[0], RESIZE_IMG[1]))])
    depth_transform = T.Compose([T.Resize((RESIZE_IMG[0], RESIZE_IMG[1]))])
    
    val_dataset = KITTI360_Fisheye_Dataset(rootdir="/home/rangganast/rangganast/dataset/KITTI-360",
                                        sequences=[0, 2, 3, 4, 5, 6, 7, 9, 10],
                                        split="val",
                                        camera_id="02",
                                        frame_step=1,
                                        n_scans=None,
                                        voxel_size=None,
                                        max_trans=[0.5],
                                        max_rot=[5.0],
                                        rgb_transform=rgb_transform,
                                        depth_transform=depth_transform,
                                        return_pcd=True,
                                        device=DEVICE)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=0, drop_last=True)

    model = Net().to(DEVICE)
    model, last_epoch, last_val_loss = load_checkpoint(LOAD_CHECKPOINT_DIR, model)

    test(
        model,
        val_loader,
        BATCH_SIZE
    )