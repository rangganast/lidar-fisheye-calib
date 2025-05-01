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

* Project the point onto a normalized coordinate system.

$$
m = \left( \frac{X_s}{Z_s + \xi}, \frac{Y_s}{Z_s + \xi}, 1 \right) = \hbar(\chi_s)
$$

* The final projection matrix incorporates the camera's intrinsic parameters.

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

TBA

#### Loss Functions
A Three loss functions are employed for the model training. The combined loss function are expressed as follows:

$$
L = \lambda_1 L_{trans} + \lambda_2 L_{rot},
$$

where $\lambda_1$ and $\lambda_2$ are the weights for each loss term.
Each loss function term is expressed as below:

* Translation Loss

TBA

* Rotation Loss (Quaternion Distance)

TBA

## Getting Started
### Requirements
TBA

### Training
TBA
```

### Evaluation
TBA

## Current Result
TBA

## Contacts
Rangga Aziz

Email:
rangganast@gmail.com

## Acknowledgments