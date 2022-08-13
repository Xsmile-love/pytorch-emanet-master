# pytorch-emanet-master
By Hua Yang and Jing Yang

[College of Electrical Engineering, Guizhou University]

This repo is the official implementation of **"Multiscale Hybrid Convolutional Deep Neural Network with Channel Attention"**


# Approach
![MC nodule](https://user-images.githubusercontent.com/78161848/178894262-2fa77c60-18e9-46d9-b715-76e5a843bb23.png)
<p align="center">
Figure 1: Overall architecture of MC module
</p>


# Installation
### Requirements
- Python 3.8+
- Pytorch 1.10+
- thop
### Our environments
- OS:Ubuntu 20.04
- CUDA:11.3
- Toolkit:Pytorch 1.11.0
- GPU:RTX3060Ti
### Other libraries
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


# Experiments
### CIFAR-10 Classification
|Networks|Param.|FLOPs|Top-1(%)|
|---|---|---|---|
|ResNet50|22.43M|1.215G|93.62|
|CBAM|24.83M|1.222G|93.43|
|SA-Net|22.43M|1.216G|93.79|
|SENet|24.83M|1.219G|95.35|
|FcaNet|24.83M|1.217G|95.49|
|ECANet|22.43M|1.217G|95.35|
|EPSANet|19.58M|1.066G|95.32|
|EMANet|20.20M|1.119G|95.61|

### PASCAL VOC2007
#### Detection with Faster R-CNN
|Backbone|Param.|FLOPs|AP|AP_50|AP_75|
|---|---|---|---|---|---|
|SENet|30.98M|351.76G|48.1|80.5|50.3|
|ResNet|28.47M|317.98G|45.7|78.3|47.8|
|FcaNet|30.98M|351.63G|49.4|82.3|52.6|
|ECANet|28.47M|351.11G|48.4|81|50.9|
|EMANet|26.13M|268.57G|53.9|84.8|59|


# Reference

[https://github.com/gbup-group/IEBN](https://github.com/gbup-group/IEBN)  
[https://github.com/bubbliiiing/faster-rcnn-pytorch](https://github.com/bubbliiiing/faster-rcnn-pytorch)  
[https://github.com/open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
