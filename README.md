# LiDAR-Camera Extrinsic Calibration
This repository provides a project about deep-learning-based targetless extrinsic calibration for LiDAR--Fisheye Camera system.

## Table of Contents
- [About the Project](#about-the-project)
    * [Problem Formulation](#problem-formulation)
    * [Data Preprocessing](#data-preprocessing)
    * [Model Architecture](#model-architecture)
- [Getting Started](#getting-started)
    * [Requirements](#requirements)
    * [Training](#training)
    * [Evaluation](#evaluation)
- [Current Result](#current-result)
- [Contacts](#contacts)
- [Acknowledgments](#acknowledgments)

## About the Project 
This project was part of the sensor fusion and computer fusion research project at [WiComAI Lab](https://wireless.kookmin.ac.kr/) of Kookmin University, Seoul, South Korea. 
The project aims to predicts the misalignment in the extrinsic parameters between the camera and LiDAR, inspired by [Spherical Convolution](hhttps://arxiv.org/pdf/2102.05013), and by leveraging a residual transformer with deformable attention 
as feature extraction network. This network also utilize cost volume correlation, inspired by [MRCNet](https://ieeexplore.ieee.org/document/10308709) as feature matching network. The deep learning network is trained 
and tested using the [KITTI360 dataset](https://www.cvlibs.net/datasets/kitti-360/).

#### Problem Formulation
Extrinsic parameters in LiDAR-Camera system is used to perform coordinate transformation of 3D poinnts from LiDAR coordinate system to camera coordinate system.
This coordinate transformation is used to perform data fusion between the two modalities:

Given a set LiDAR points $P_L: [X_L \quad Y_L \quad Z_L]^{T}$ and 
LiDAR points set in camera coordinate frame $P_C: [X_C \quad Y_C \quad Z_C]^{T}$, 
the coordinate transformation of a single LiDAR point
is expressed as:

$$
P_C = \begin{bmatrix} R & | & t \end{bmatrix} \cdot P_L = T \cdot P_L,
$$

where $T$, $R$, and $t$ are the extrinsic parameter matrix, rotation matrix, and translation vector of the LiDAR
camera system, respectively.

The network aims to correct the deviation in extrinsic parameters accumulated during vehicles operations, where this deviation can be expressed as:

$$ 
T_{actual} =  \Delta T \cdot T_{known}, 
$$

where $T_{actual}$ is the actual extrinsic parameter after deviation is considered, $T_{known}$ us the initial extrinsic parameter without the deviation considered, 
and $\Delta T$ is the extrinsic parameter deviation represented in a rigid transformation. 
The network predicts the value of $\Delta T$ to obtain the calibrated extrinsic parameter, which expressed as:

$$
T_{actual} = \Delta T_{predicted}^{-1} \cdot \Delta T \cdot T_{known}.
$$

Assuming that the prediction is highly accurate, the deviation term will cancel out.

#### Data Preprocessing
The network uses RGB images and depth images from 2D projections of LiDAR point clouds. 
Depth images are created by projecting LiDAR point to image plane using [Mei and Rives camera model](https://ieeexplore.ieee.org/document/4209702). Here are some steps:

* Project 3D world points in the mirror frame onto a unit sphere.
$$
\mathbf{(\chi)_F}_m \rightarrow \mathbf{(\chi_s)_F}_m = \frac{\chi}{\|\chi\|} = (X_s, Y_s, Z_s)
$$

* Points are transformed to a new reference frame based on the center of the new frame.
$$
C_p = (0,0,\xi),\mathbf{(\chi_s)_F}_m \rightarrow \mathbf{(\chi_s)_F}_p = (X_s, Y_s, Z_s + \xi)
$$

\item Project the point onto a normalized coordinate system.
$$
m = \left( \frac{X_s}{Z_s + \xi}, \frac{Y_s}{Z_s + \xi}, 1 \right) = \hbar(\chi_s)
$$

\item The final projection matrix incorporates the camera's intrinsic parameters.
$$
p = K m =
\begin{bmatrix}
f_1 \eta & f_1 \eta \alpha & v_0 \\
0 & f_2 \eta & v_0 \\
0 & 0 & 1
\end{bmatrix} m = k(m)
$$

where $f_x$, $f_y$, $c_x$, and $c_y$ are the intrinsic parameters of the camera.


#### Model Architecture
| ![architecture](./figures/transcalib_architecture_rev.png) |
|:--:| 
| *Architecture of the Extrinsic Calibration Deep Network* |

The model consists of two branches of feature extraction, each for RGB image and LiDAR depth image. 
The two branches use the same architecture of [EfficientNetV2](https://arxiv.org/abs/2104.00298) without the fully-connected layers 
and different input channel for each branch (3 channels for RGB and 1 channel for depth).

The resulting feature maps from both branches is the concatenated and are processed in the feature matching network.
The feature matching network leverages CSA transformer from [Lite Vision Transformer](https://arxiv.org/abs/2112.10809)
to find corresponding features from both modalities.

The feature mathcing result is then fed to the fully connected layers where $1 \times 3$ translation vectors and
$1 \times 4$ rotation vectors (in quaternions) are predicted.

#### Loss Functions
A Three loss functions are employed for the model training. The combined loss function are expressed as follows:

$$
L = \lambda_1 L_{trans} + \lambda_2 L_{rot} + \lambda_3 L_{\text{PCD}},
$$

where $\lambda_1$, $\lambda_2$, and $\lambda_3$ are the weights for each loss term.
Each loss function term is expressed as below:

* Translation Loss

$$
L_{trans}(t_{pred}, t_{gt}) = \frac{1}{n} \sum_{i}^{n} \text{smoothL1}(t_{pred\ i} - t_{gt\ i}),
$$

$$
\text{smoothL1}(x) =
\begin{cases} 
\frac{1}{2} x^2, & \text{if } |x| < 1 \\
|x| - \frac{1}{2}, & \text{otherwise}
\end{cases}
$$

* Rotation Loss (Quaternion Distance)

$$
L_{rot}(q_{pred}, q_{gt}) = D_a(q_{gt} * \text{inv}(q_{pred})),
$$

$$
D_a(m) = \text{atan2}\left(\sqrt{b^2_m + c^2_m + d^2_m}, \, |a_m|\right),
$$

* Point Cloud Distance Loss (Chamfer Distance)

$$
L_{\text{PCD}}(P_{pred}, P_{gt}) = \sum_{x \in P_{pred}} \min_{y \in P_{gt}} \| x - y \|^2 + \sum_{y \in P_{gt}} \min_{x \in P_{pred}} \| y - x \|^2
$$

where $P_{pred}$ and $P_{gt}$ are the ground truth point cloud data and point cloud data calibrated with predicted extrinsic parameters, respectively.

## Getting Started
### Requirements
First, download [KITTI Odometry dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) on your setup.
After downloading the dataset, modify the path to the dataset in the training file (`train_with_cometml.py` or `train.py`)
and the testing file (`test_continuous.py` or `test_iterative.py`):
```
DATASET_FILEPATH = "/path/to/kitti/odometry/dataset"
```

This project was developed on an environment consisting of:
* Python 3.10.12
* NVIDIA RTX 3090Ti GPU with CUDA v11.8 and CuDNN v11.8
* Install dependencies using the syntax below:
```
pip install -r requirements.txt
```

### Training
To train without using Comet ML, run this on your terminal:
```
python train.py
```
To train using Comet ML, run:
```
python train_with_cometml.py
```
Before using `train_with_cometml.py`, modify these lines:
```
experiment = Experiment(
    api_key = YOUR_COMETML_API, # modify to your COMET ML API
    project_name = "your_project_name", # modify to your project name
    workspace = "your_workspace_name", # modify to your workspace name
    ...
)
```

### Evaluation


## Current Result


## Contacts
Miftahul Umam

Email:
miftahul.umam14@gmail.com

## Acknowledgments