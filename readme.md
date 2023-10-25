This project provides the implementations of some data augmentation methods, regularization methods, online Knowledge distillation and Self-Knowledge distillation methods.

## Installation

### Requirements

Ubuntu 18.04 LTS

Python 3.8 ([Anaconda](https://www.anaconda.com/) is recommended)

CUDA 11.1

PyTorch+torchvision

please install python packages:
```
pip install -r requirements.txt
```

## Perform experiments on image classification
### Dataset
CIFAR-100 : [download](http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)

baseline：
```
python main.py \
    --dataset CIFAR-100 \
    --data_folder [your dataset path]\
    --kd-method cross_entropy \
    --checkpoint-dir [your checkpoint path]
```

CIMGS-KD method：
```
python main.py \
    --dataset CIFAR-100 \
    --data_folder [your dataset path]\
    --kd-method CIMGS-KD \
    --checkpoint-dir [your checkpoint path]
```

<table>
	<tr>
	    <td>Method</td>
      <td>Accuracy(%)</td>  
	</tr >
	<tr >
    <td>Baseline</td>
	    <td>76.38</td>
	</tr>
    <td>CIMGS-KD</td>
	    <td>81.82</td>
	</tr>
  <tr >
</table>

CUB-200-2011 : [download](https://www.vision.caltech.edu/datasets/cub_200_2011/)

baseline：
```
python main.py \
    --arch resnet18 \
    --dataset CUB200 \
    --data_folder [your dataset path]\
    --kd-method cross_entropy \
    --checkpoint-dir [your checkpoint path]
```

CIMGS-KD method：
```
python main.py \
    --arch resnet18 \
    --dataset CUB200 \
    --data_folder [your dataset path]\
    --kd-method CIMGS-KD \
    --checkpoint-dir [your checkpoint path]
```

<table>
	<tr>
	    <td>Method</td>
      <td>Accuracy(%)</td>  
	</tr >
	<tr >
    <td>Baseline</td>
	    <td>62.70</td>
	</tr>
    <td>CIMGS-KD</td>
	    <td>73.68</td>
	</tr>
  <tr >
</table>


stanford dogs: [download](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html)

baseline：
```
python main.py \
    --arch resnet18 \
    --dataset STANFORD120 \
    --data_folder [your dataset path]\
    --kd-method cross_entropy \
    --checkpoint-dir [your checkpoint path]
```

CIMGS-KD method：
```
python main.py \
    --arch resnet18 \
    --dataset STANFORD120 \
    --data_folder [your dataset path]\
    --kd-method CIMGS-KD \
    --checkpoint-dir [your checkpoint path]
```

<table>
	<tr>
	    <td>Method</td>
      <td>Accuracy(%)</td>  
	</tr >
	<tr >
    <td>Baseline</td>
	    <td>68.14</td>
	</tr>
    <td>CIMGS-KD</td>
	    <td>73.48</td>
	</tr>
  <tr >
</table>

ImageNet: [download](https://www.image-net.org/download.php)

baseline：
```
python main.py \
    --arch resnet18 \
    --dataset imagenet \
    --data_folder [your dataset path]\
    --kd-method cross_entropy \
    --checkpoint-dir [your checkpoint path]
```

CIMGS-KD method：
```
python main.py \
    --arch resnet18 \
    --dataset imagenet \
    --data_folder [your dataset path]\
    --kd-method CIMGS-KD \
    --checkpoint-dir [your checkpoint path]
```

<table>
	<tr>
	    <td>Method</td>
      <td>Accuracy(%)</td>  
	</tr >
	<tr >
    <td>Baseline</td>
	    <td>76.32</td>
	</tr>
    <td>CIMGS-KD</td>
	    <td>79.38</td>
	</tr>
  <tr >
</table>