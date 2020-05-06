Attend in groups: a weakly-supervised deep learning framework for learning from web data


***If you use this code in your research, please cite our paper:***

```
@InProceedings{Zhuang_2017_CVPR,
author = {Zhuang, Bohan and Liu, Lingqiao and Li, Yao and Shen, Chunhua and Reid, Ian},
title = {Attend in Groups: A Weakly-Supervised Deep Learning Framework for Learning From Web Data},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {July},
year = {2017}
}
```

## Dataset
We provide the WebCars dataset here 
https://mega.nz/file/rZpTVZbD#tleS-mP7XnIO2-_RoxYcO_hUfveURh5XkdGUhQ8fLAM

## Code
The code are written using Lasagne.

__utils.py__: provide necessary functions  
__vggnet.py__: define network structure  
__train.py__: main file, implementing training and testing   
__config.yaml__: define the necessary hyperparameters (e.g., data directory, bag size, GPU), please modify this file  
__./pretrained_model__: the pretrained VGG16 model on ImageNet  
__img_mean.npy__: mean file for data preprocessing  


## Training

```
python train.py

```


## Copyright

Copyright (c) Bohan Zhuang. 2017

** This code is for non-commercial purposes only. For commerical purposes,
please contact Chunhua Shen <chhshen@gmail.com> **

This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
