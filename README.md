# License Plate Location 

Design algorithms to realize license plate recognition：

- Traditional processing method based on shape filtering
- Traditional processing method based on color shape filtering
- Deep learning method for target detection model, Yolo

## Background

The key to **license plate recognition** (LPR) lies in **license plate location**, character segmentation, and character recognition. The accuracy of the license plate location directly determines the subsequent character segmentation and recognition effect, which is the most critical step in the license plate recognition technology.

According to the different license plate features, **traditional license plate detection methods** are divided into license plate detection methods based on edge detection [1], license plate detection methods based on character features [2], license plate detection methods based on color features [3] and texture-based license plate detection methods. There are four feature-based license plate detection methods [4].

In recent years, the field of artificial intelligence has developed rapidly. Deep learning methods have achieved outstanding results in the field of target detection, which has made breakthroughs in license plate detection research. Object detection algorithms represented by YOLO[5] series, SSD algorithm[6], and Faster R-CNN[7] have achieved extensive and far-reaching influence.

## Structure

```
-License Plate Recognition
|---models
	|---ColorLocation.py # traditional color-based method
	|---PlateLocation.py  # traditional shape-based method
	|---YOLO_detection.py # Yolo
	|---LPRNET.py
|---utils
	|---DateLoader.py
	|---utils.py
	|---Others rely on external code
|---main.py 		# Select method (All 3) 
|---params.py       # Set params
|---train.py
```

## Requirements

- Python 3.7 
- Torch 1.8.0 
- Numpy 1.18.0 
- Opencv-python4.5.4.60 
- Cpython 0.28.5 
- Glob 0.6

## Experiment Results

<img src="https://github.com/supergirl-os/License-Plate-Recognition/raw/main/res2.png" alt="Aaron Swartz" style="zoom:67%;" />

Figure 1 The first row is the IoU of recognition results, the second row is the mask map, and the third row is the license plate detection map. （Traditional method based on both color and shape）



<img src="https://github.com/supergirl-os/License-Plate-Recognition/raw/main/res3.png" alt="Aaron Swartz" style="zoom:67%;" />

Figure 2 YOLO for locating license plates on images with different features

## References

[1] ZHENG D N, ZHAO Y N, WANG J X. An efficient method of license plate location[J].Pattern recognition letters, 2005, 26(15): 2431-2438. 

[2] ZHOU W G, LI H Q, LU Y J, et al. Principal visual word discovery for automatic license plate detection[J].IEEE Transactions on Image Processing A Publication of the IEEE Signal Processing Society, 2012, 21(9):4269-79. 

[3] ABOLGHASEMI V, AHMADYFARD A. An edge-based color-aided method for license plate detection[J]. Image and Vision Computing, 2009, 27(8): 1134-1142. 

[4] YU S, LI B, ZHANG Q, et al. A novel license plate location method based on wavelet transform and EMD analysis[J]. Pattern Recognition, 2015, 48(1): 114-125. 

[5] REDMON J, DIVVALA S, et al. You only look once: unified, real-time object detection[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. Piscataway, NJ: IEEE, 2016:779-788. 

[6] LIU W, ANGUELOV D, ERHAN D, et al. SSD: single shot multi box detector[C]//Proceedings of European Conference on Computer Vision. Heidelberg: Springer, 2016: 21-37. 

[7] REN S Q, HE K M, GIRSHICK R, et al. Faster R-CNN: towards real-time object detection with region proposal networks[C]//Proceedings of Neural Information Processing System. Cambridge: MIT Press, 2015: 91-99.