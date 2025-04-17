from comet_ml import Experiment

import os
import time
from datetime import datetime
from types import SimpleNamespace as sns
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms as T

from dataset.dataset import KITTI360_Fisheye_Dataset
from models.model import Net

def validate(model, loader, training_cfg, val_step):
    val_loss_epoch = 0
    
    with torch.no_grad():
        model.eval()
        for batch_data in tqdm(loader, desc="Validation"):
            val_step += 1
            
            rgb_img = batch_data["img"].to(DEVICE)
            depth_img = batch_data["depth_img_error"]
            delta_q_gt = batch_data["delta_q_gt"].to(DEVICE)
            delta_t_gt = batch_data["delta_t_gt"].to(DEVICE)
            
            # run prediction
            delta_q_pred, delta_t_pred = model(rgb_img, depth_img)

            val_loss = get_loss(delta_q_pred, delta_t_pred, delta_q_gt, delta_t_gt)
            
            val_loss_epoch += val_loss.item()

            experiment.log_metric("val step", val_step)
            experiment.log_metric("validation batch loss", val_loss.item())
                
    val_loss_epoch /= len(loader)

    return val_loss_epoch

def train(model,
          train_loader,
          val_loader=None,
          training_cfg=None,
          optimizer=None,
          scheduler=None,
          last_epoch=None,
          last_best_loss=None):

    print("last epoch:", last_epoch)
    print("last best loss:", last_best_loss)

    if last_best_loss is None:
        best_val_loss = 1500000000.0
    else:
        best_val_loss = last_best_loss

    if last_best_loss is None:
        best_train_loss = 1500000000.0
    else:
        best_train_loss = last_best_loss

    train_step = 0
    val_step = 0

    print("training is running...")
    for i in range(last_epoch, training_cfg.n_epochs):
        start_epoch = time.time()
        model.train()
        print("epoch: ", i+1)
        loss_epoch = 0
        
        for batch_data in tqdm(train_loader, desc="Training"):      
            optimizer.zero_grad()
            train_step += 1
            
            rgb_img = batch_data["img"].to(DEVICE)
            depth_img = batch_data["depth_img_error"]
            delta_q_gt = batch_data["delta_q_gt"].to(DEVICE)
            delta_t_gt = batch_data["delta_t_gt"].to(DEVICE)
            
            # run prediction
            delta_q_pred, delta_t_pred = model(rgb_img, depth_img)

            loss = get_loss(delta_q_pred, delta_t_pred, delta_q_gt, delta_t_gt)
            
            loss.backward()
            
            if GRAD_CLIP is not None:
                clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer.step()

            loss_epoch += loss.item()

            experiment.log_metric("train step", train_step)
            experiment.log_metric("training batch loss", loss.item())
            
        end_epoch = time.time()
        print(f"epoch time: {end_epoch - start_epoch}")
        
        loss_epoch /= len(train_loader)

        experiment.log_metric("training epoch loss", loss_epoch)
            
        print("=================")
        print("total loss: ", loss_epoch)
        print("=================")
        print()

        
        print("validation is starting...")
        val_loss = validate(model, val_loader, training_cfg, val_step)
        experiment.log_metric("validation epoch loss", val_loss)
        
        scheduler.step(val_loss)

        print()

        print("=================")
        print("val_loss: ", val_loss)
        print("=================")
        print()

        experiment.log_metric("learning rate", optimizer.param_groups[0]['lr'])
        print("current learning rate:", str(optimizer.param_groups[0]['lr']))
        
        print("saving model...")
        # save checkpoint with the best validation score
        if val_loss < best_val_loss:
            print("best val loss achieved")
            checkpoint = {"state_dict": model.state_dict(),
                          "optimizer:": optimizer.state_dict(),
                          "epoch": i+1,
                          "loss": val_loss}
            save_checkpoint(checkpoint, filename=BEST_VAL_CHECKPOINT_DIR)
            best_val_loss = val_loss

        # save checkpoint with the best training score
        if loss_epoch < best_train_loss:
            print("best train loss achieved")
            checkpoint = {"state_dict": model.state_dict(),
                          "optimizer:": optimizer.state_dict(),
                          "epoch": i+1,
                          "loss": loss_epoch}
            save_checkpoint(checkpoint, filename=BEST_TRAIN_CHECKPOINT_DIR)
            best_train_loss = loss_epoch

def save_checkpoint(state, filename="ViT_checkpoint.pth.tar"):

    print("=> Saving checkpoint")
    print()
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model):
    print("=> Loading checkpoint")
    print()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    last_epoch = checkpoint["epoch"]
    last_epoch_loss = checkpoint["loss"]

    return model, last_epoch, last_epoch_loss

def get_loss(q1, t1, qq_gt, t_gt):
    l1_q_norm = q1 / (torch.sqrt(torch.sum(q1 * q1, dim=-1, keepdim=True) + 1e-10) + 1e-10)
    l1_loss_q = torch.mean(torch.sqrt(torch.sum((qq_gt - l1_q_norm) * (qq_gt - l1_q_norm), dim=-1, keepdim=True) + 1e-10))
    l1_loss_x = torch.mean(torch.sqrt((t1 - t_gt) * (t1 - t_gt) + 1e-10))
    loss_sum = l1_loss_x + l1_loss_q

    return loss_sum

if __name__ == "__main__":
    torch.set_printoptions(precision=10)
    torch.multiprocessing.set_start_method('spawn')

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"

    DEVICE = "cuda:0"
    NUM_WORKERS = 0 # 6
    PIN_MEMORY = False
    GRAD_CLIP = 1.0
    LOAD_MODEL = False
    VERSION = "v1"

    try:
        os.makedirs("checkpoint_weights", exist_ok=False)
        os.makedirs("checkpoint_weights/run", exist_ok=False)
        os.makedirs("checkpoint_weights/run/", exist_ok=False)
        os.makedirs(f"checkpoint_weights/run/{VERSION}", exist_ok=False)
    except:
        pass

    LOAD_CHECKPOINT_DIR = f"checkpoint_weights/run/{VERSION}/360_Calib_train_model_242312.pth.tar"
    BEST_TRAIN_CHECKPOINT_DIR = f"checkpoint_weights/run/{VERSION}/360_Calib_train_model_242312.pth.tar"
    BEST_VAL_CHECKPOINT_DIR = f"checkpoint_weights/run/{VERSION}/360_Calib_val_model_242312.pth.tar"
    BATCH_SIZE = 1 # 16
    MODEL_NAME = "360_Calib_model_242312"
    MODEL_DATE = datetime.now().strftime('%Y%m%d_%H%M%S')
    RESIZE_IMG = [350, 350]

    experiment = Experiment(
        api_key=os.environ.get("COMETML_API_KEY"),
        project_name="360-calib",
        workspace="krishnayoga",
        auto_metric_logging=True,
        auto_param_logging=True,
        auto_histogram_weight_logging=False,
        auto_histogram_gradient_logging=False,
        auto_histogram_activation_logging=False,
    )

    experiment.set_name(f'{MODEL_NAME}_{MODEL_DATE}')

    training_cfg = {
        'n_epochs': 1000,
        'learning_rate':1e-4,
        'min_lr': 0.00001,
        'momentum': 0.8,
        'loss_threshold': 0.0001,
        'device': DEVICE,
        'batch_size': BATCH_SIZE,
    }
    
    sns_training_cfg = sns(**training_cfg)
    
    rgb_transform = T.Compose([T.Resize((RESIZE_IMG[0], RESIZE_IMG[1]))])
    depth_transform = T.Compose([T.Resize((RESIZE_IMG[0], RESIZE_IMG[1]))])
    
    train_dataset = KITTI360_Fisheye_Dataset(rootdir="/home/rangganast/rangganast/dataset/KITTI-360",
                                        sequences=[2, 3, 4, 5, 6, 7, 9],
                                        split="train",
                                        camera_id="02",
                                        frame_step=1,
                                        n_scans=None,
                                        voxel_size=None,
                                        max_trans = [1.5, 0.5, 0.1, 0.1],
                                        max_rot = [20, 10, 5, 1],
                                        rgb_transform=rgb_transform,
                                        depth_transform=depth_transform,
                                        return_pcd=False,
                                        device=DEVICE)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS, drop_last=True)

    val_dataset = KITTI360_Fisheye_Dataset(rootdir="/home/rangganast/rangganast/dataset/KITTI-360",
                                        sequences=[0],
                                        split="val",
                                        camera_id="02",
                                        frame_step=1,
                                        n_scans=None,
                                        voxel_size=None,
                                        max_trans = [1.5, 0.5, 0.1, 0.1],
                                        max_rot = [20, 10, 5, 1],
                                        rgb_transform=rgb_transform,
                                        depth_transform=depth_transform,
                                        return_pcd=False,
                                        device=DEVICE)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS, drop_last=True)
    
    model = Net().to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=sns_training_cfg.learning_rate, weight_decay=1e-4, betas=(0.9, 0.999))

    if LOAD_MODEL:
        model, last_epoch, last_val_loss = load_checkpoint(LOAD_CHECKPOINT_DIR, model)
    else:
        last_epoch, last_val_loss = 0, None
        
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=10, verbose=True, eps=1e-50)

    train(model=model,
          train_loader=train_loader,
          val_loader=val_loader,
          training_cfg=sns_training_cfg,
          optimizer=optimizer,
          scheduler=scheduler,
          last_epoch=last_epoch,
          last_best_loss=last_val_loss)
    