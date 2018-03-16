# COMP5421 Project 2
This project is an implementation of the sliding window detector of Dalal and Triggs 2005.
## Algorithm
### Load data and extract feature
We calculate HOG feature of each face image and convert the result to a vecter. Then we stack features of all the images to a matrix. 

We augment the face images in the following way
1. Use left-right flipped copies
2. Slightly zoom in each image and randomly crop a 36*36 patch. 

For non-face scenes, we also ranodmly crop a 36*36 patches and calculate HOG features. 

### SVM
We use the provided SVM trainer to obtian a linear classifier. 

### Sliding window detector
We do the following for each test scene. 

1. Resize the image with different scales
2. Calculate HOG
3. Get a window with the same size as training faces
4. Calculate the response of linear classifier
6. Record this window if the response is larger than the threshold
7. use non-max suppression to select the best bounding box from the ones around a single face

## Parameters
In data augmentation of positive samples, we randomly choose a number from 1.0 to 1.1 as the resize scale and generate 8 samples for one face picture. We set the number of negative samples to be 50000. The weight decay of SVM is 1e-4. We use -0.5 as the threshold in the detector. And we use resize scales from 0.1 to 1.2 with step 0.05. 

## Result
The best average precision we have achieved is 93.2. But this detector will report many false positives. 

![AP](https://i.imgur.com/3lhf8oOm.png)![weight](https://i.imgur.com/wIOMuR4m.png)

![result](https://i.imgur.com/qEFcPbX.png)

Data augmentation slightly improve the accuracy. We test the detector by disabling the data augmentation and changing the number of negative samples to 10000 to preserve similar ratio of positive and negative data. 

![AP_no_aug](https://i.imgur.com/Lpn87com.png)

We use 2.0 as threshold on the extra test images. 

![Imgur](https://i.imgur.com/1Nv4jph.png)![Imgur](https://i.imgur.com/LABRHee.png)![Imgur](https://i.imgur.com/pn3CDsT.png)![Imgur](https://i.imgur.com/0vAOGBg.png)![Imgur](https://i.imgur.com/f90Si75.png)![Imgur](https://i.imgur.com/Se0P9PR.png)![Imgur](https://i.imgur.com/tFqfmKc.png)

## Contribution
Luqi Wang and Jingyang Zhang contributed equally to this project.