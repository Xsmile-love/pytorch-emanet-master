# pytorch-emanet-master
By Hua Yang and Jing Yang

[College of Electrical Engineering, Guizhou University]

This repo is the official implementation of **"Multiscale Hybrid Convolutional Deep Neural Network with Channel Attention"**

# Install the required libraries

```pip/conda install ...```
```
scipy==1.7.3
numpy==1.20.3
matplotlib==3.5.1
opencv_python==4.5.5.64
torch==1.11.0
torchvision==0.11.0
tqdm==4.63.0
Pillow==9.0.1
h5py==2.10.0
```
# Evaluation
To evaluate a pre-trained emanet50 on CIFAR10 val with a single GPU run:

```
 python main.py -a emanet50 -e --resume /checkpoint/emanet/ckpt.pth
```

# Approach
![MC nodule](https://user-images.githubusercontent.com/78161848/178894262-2fa77c60-18e9-46d9-b715-76e5a843bb23.png)
<p align="center">
Figure 1: Overall architecture of MC module
</p>


# Mini-ImageNet Classification
![Figure](https://user-images.githubusercontent.com/78161848/178894589-6852cd21-4c37-4086-aea0-c271ff810d61.png)
<p align="center">
Figure 2: Comparisons of recently SOTA attention models on mini-ImageNet, using ResNets as backbones, in terms of accuracy, network parameters, and FLOPs. The size of circles indicates the FLOPs
</p>


# Reference

[https://github.com/gbup-group/IEBN](https://github.com/gbup-group/IEBN)  
[https://github.com/bubbliiiing/faster-rcnn-pytorch](https://github.com/bubbliiiing/faster-rcnn-pytorch)  
[https://github.com/open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
