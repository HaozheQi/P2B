# P2B: Point-to-Box Network for 3D Object Tracking in Point Clouds

## Introduction

This repository is code release for our CVPR 2020 paper. In this repository, we provide P2B model implementation (with Pytorch) as well as data preparation, training and evaluation scripts on KITTI tracking dataset.

## Installation

* Install ``python 3.6``.

* Install dependencies
 ```
 pip install -r requirements.txt
 ```

* Building `_ext` module
 ```
python setup.py build_ext --inplace
 ```
 
* Download the dataset from [KITTI Tracking](http://www.cvlibs.net/datasets/kitti/eval_tracking.php).

	You will need to download the data for
	[velodyne](http://www.cvlibs.net/download.php?file=data_tracking_velodyne.zip), 
	[calib](http://www.cvlibs.net/download.php?file=data_tracking_calib.zip) and
	[label_02](http://www.cvlibs.net/download.php?file=data_tracking_label_2.zip).
	Place the 3 folders in the same parent folder.

## Training and evaluating

To train a new P2B model on KITTI data:
```
python train_tracking.py --data_dir=<kitti data path>
```

To test a new P2B model on KITTI data:
```
python test_tracking.py --data_dir=<kitti data path>
```

Optionally, you can change other arguments in the code. 

## Acknowledgements

Thank Erik Wijmans for his implementation of [PointNet++](https://github.com/erikwijmans/Pointnet2_PyTorch) in Pytorch.

Thank Giancola for his implementation of [SC3D](https://github.com/SilvioGiancola/ShapeCompletion3DTracking).

Thank Charles R. Qi for his implementation of [Votenet](https://github.com/facebookresearch/votenet).

They all help and inspire me a lot in completing this work!
