# Bidirectional Joint Estimation of Optical Flow and Scene Flow with Gaussian-Guided Attention

This project provides the official implementation of 'Bidirectional Joint Estimation of Optical Flow and Scene Flow with Gaussian-Guided Attention'.

## Abstract
Optical flow and scene flow from images and point clouds serve to jointly estimate the motion field, which has extensive applications in robotics. Using two complementary modalities, the fusion estimation process often neglects the fact that visual images inherently contain more information, the reason being that visual information exhibits dense characteristics in perception, whereas point clouds sample the three-dimensional space sparsely and non-uniformly. In order to further exploit the fine-grained visual information and the complementarity between these two modalities, we propose a method for the bidirectional joint estimation of optical flow and scene flow with Gaussian-guided attention. This method leverages the Gaussian attention to further exploit the fine-grained texture information inherent in the dense visual data. Moreover, through bidirectional fusion mechanisms, the smoothness priors of the two modalities, mainly constrained in the 2D channel, are further fused iteratively through the Gaussian attention for capturing the matching-prior knowledge. Since the proposed method can utilize the information possessed by both modalities, it enables better joint estimation of the optical flow and scene flow. Experimental results show that the proposed method can achieve competitive performance on public datasets and prevails on the FlyingThings3D and KITTI datasets.

## Environment

Create a PyTorch environment using `conda`:

```
conda create -n camliga python=3.7
conda activate camliga
conda install pytorch==1.10.2 torchvision==0.11.3 cudatoolkit=11.3 -c pytorch
```

Install mmcv and mmdet:

```
pip install openmim
mim install mmcv-full==1.4.0
mim install mmdet==2.14.0
```

Install other dependencies:

```
pip install opencv-python open3d tensorboard hydra-core==1.1.0
pip install einops
pip install easydict
pip install scipy
pip install timm
pip install natten
```

Compile CUDA extensions for faster training and evaluation:

```
cd models/csrc
python setup.py build_ext --inplace
```

Download the ResNet-50 pretrained on ImageNet-1k:

```
wget https://download.pytorch.org/models/resnet50-11ad3fa6.pth
mkdir pretrain
mv resnet50-11ad3fa6.pth pretrain/
```

NG-RANSAC is also required if you want to evaluate on KITTI. Please follow [https://github.com/vislearn/ngransac](https://github.com/vislearn/ngransac) to install the library.

## Demo

Then, run the following script to launch a demo of estimating optical flow and scene flow from a pair of images and point clouds:

```
python demo.py --model camliga --weights /path/to/checkpoint.pt
```

## Evaluation

### FlyingThings3D

First, download and preprocess the dataset (see `preprocess_flyingthings3d_subset.py` for detailed instructions):

```
python preprocess_flyingthings3d_subset.py --input_dir your_file_path/data/flyingthings3d_subset
```

Then, download the pretrained weights [ft3d.pt](https://drive.google.com/file/d/1KbGQagoVlPwVf94p62iqVQ9oituMXOUx/view?usp=drive_link) and save it to `checkpoints/ft3d.pt`.

Now you can reproduce the results :

```
python eval_things.py testset=flyingthings3d_subset model=camliga ckpt.path=checkpoints/ft3d.pt
```

### KITTI

First, download the following parts:

* Main data: [data_scene_flow.zip](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip)
* Calibration files: [data_scene_flow_calib.zip](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow_calib.zip)
* Disparity estimation (from GA-Net): [disp_ganet.zip](https://drive.google.com/file/d/1ieFpOVzqCzT8TXNk1zm2d9RLkrcaI78o/view?usp=sharing)
* Semantic segmentation (from DDR-Net): [semantic_ddr.zip](https://drive.google.com/file/d/1dVSJeE9BBmVv2rCe5TR0PVanEv2WzwIy/view?usp=sharing)

Unzip them and organize the directory as follows:

```
datasets/kitti_scene_flow
├── testing
│   ├── calib_cam_to_cam
│   ├── calib_imu_to_velo
│   ├── calib_velo_to_cam
│   ├── disp_ganet
│   ├── flow_occ
│   ├── image_2
│   ├── image_3
│   ├── semantic_ddr
└── training
    ├── calib_cam_to_cam
    ├── calib_imu_to_velo
    ├── calib_velo_to_cam
    ├── disp_ganet
    ├── disp_occ_0
    ├── disp_occ_1
    ├── flow_occ
    ├── image_2
    ├── image_3
    ├── obj_map
    ├── semantic_ddr
```

Then, download the pretrained weights [kitti.pt](https://drive.google.com/file/d/15omp0R7W3iRBABC-prLcw599mnC-ovCI/view?usp=drive_link) and save it to `checkpoints/kitti.pt`.


```
python kitti_submission.py testset=kitti model=camliga ckpt.path=checkpoints/kitti.pt
```

To reproduce the results **with** rigid background refinement, you need to further refine the background scene flow:

```
python refine_background.py
```

Results are saved to `submission/testing`. The initial non-rigid estimations are indicated by the `_initial` suffix.

### Sintel

First, download the flow dataset from: http://sintel.is.tue.mpg.de and the depth dataset from https://sintel-depth.csail.mit.edu/landing

Unzip them and organize the directory as follows:

```
datasets/sintel
├── depth
│   ├── README_depth.txt
│   ├── sdk
│   └── training
└── flow
    ├── bundler
    ├── flow_code
    ├── README.txt
    ├── test
    └── training
```

Then, download the pretrained weights [ft3d_60E.pt](https://drive.google.com/file/d/1MBZj4A04U7oEbMWa_7TSk6GGx5H6sr9x/view?usp=drive_link) and save it to `checkpoints/ft3d_60E.pt`.

Now you can reproduce the results:

```
python eval_sintel.py testset=sintel model=camliga ckpt.path=checkpoints/ft3d_60E.pt
```

## Training

### FlyingThings3D

> You need to preprocess the FlyingThings3D dataset before training (see `preprocess_flyingthings3d_subset.py` for detailed instructions).

Train on FlyingThings3D:

```
python train.py trainset=flyingthings3d_subset valset=flyingthings3d_subset model=camliga
```

### KITTI

Finetune the model on KITTI using the weights trained on FlyingThings3D:

```
python train.py trainset=kitti valset=kitti model=camliga ckpt.path=checkpoints/ft3d.pt
```

## Acknowledgement

The code is built based on [CamLiFlow](https://github.com/MCG-NJU/CamLiFlow). We thank the authors for their contributions.
