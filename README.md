Attention as Activation
==============

MXNet/Gluon code for "Attention as Activation"

What's in this repo so far:

 * Code for CIFAR-10 and CIFAR-100 experiments with a varying network depth
 * Code for ImageNet experiments
 
## Requirements
 
Install [MXNet](https://mxnet.apache.org/) and [Gluon-CV](https://gluon-cv.mxnet.io/):
  
```
pip install --upgrade mxnet-cu100 gluoncv
```

## Experiments 

### ImageNet

Training script:
```python
python train_imagenet.py --mode hybrid --lr 0.075 --lr-mode cosine --num-epochs 160 --batch-size 128 --num-gpus 2 -j 48 --warmup-epochs 5 --dtype float16 --use-rec --last-gamma --no-wd --label-smoothing --save-dir params_resnet50_v1b_ChaATAC_2 --logging-file resnet50_v1b_ChaATAC_2.log --r 2 --act-layers 2
```

The trained model params and training log are in `./params`


| Architecture               | GFlops  | Params  | top-1 err.  | top-5 err.  |
| --------                   | ------- | ------- | ----------- | ----------- |
| ResNet-50 [[1]](#1)        | 3.86    | 25.6M   | 23.30       | 6.55        |
| SE-ResNet-50 [[2]](#2)     | 3.87    | 28.1M   | 22.12       | 5.99        |
| AA-ResNet-50 [[3]](#3)     | 8.3     | 25.8M   | 22.30       | 6.20        |
| FA-ResNet-50 [[4]](#4)     | 7.2     | 18.0M   | 22.40       | /           |
| GE-𝜽^+-ResNet-50 [[5]](#5) | 3.87    | 33.7M   | 21.88       | 5.80        |
| ATAC-ResNet-50 (ours)      | 4.4     | 28.0M   | 21.41       | 6.02        |    |


### CIFAR-10 and CIFAR-100

Training script:
```python
python train_cifar.py --gpus 0 --num-epochs 400 --mode hybrid -j 32 --batch-size 128 --wd 0.0001 --lr 0.2 --lr-decay 0.1 --lr-decay-epoch 300,350 --dataset cifar100 --model atac --act-type ChaATAC --useReLU --r 2 --blocks 3
```

<!--![](https://raw.githubusercontent.com/YimianDai/imgbed/master/github/atac/atac_cifar100_activation_c_1.png)-->

<img src=https://raw.githubusercontent.com/YimianDai/imgbed/master/github/atac/atac_cifar100_activation_c_1.png width=25%>

## References

<a id="1">[1]</a> 
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun:
Deep Residual Learning for Image Recognition. CVPR 2016: 770-778

<a id="2">[2]</a> 
Jie Hu, Li Shen, Gang Sun:
Squeeze-and-Excitation Networks. CVPR 2018: 7132-7141

<a id="3">[3]</a> 
Irwan Bello, Barret Zoph, Quoc Le, Ashish Vaswani, Jonathon Shlens:
Attention Augmented Convolutional Networks. ICCV 2019: 3285-3294

<a id="4">[4]</a> 
Niki Parmar, Prajit Ramachandran, Ashish Vaswani, Irwan Bello, Anselm Levskaya, Jon Shlens:
Stand-Alone Self-Attention in Vision Models. NeurIPS 2019: 68-80

<a id="5">[5]</a> 
Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Andrea Vedaldi:
Gather-Excite: Exploiting Feature Context in Convolutional Neural Networks. NeurIPS 2018: 9423-9433
