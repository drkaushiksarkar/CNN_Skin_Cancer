# CNN_Skin_Cancer
<<<<<<< HEAD

![Hero](assets/hero.png)

[![CI](https://img.shields.io/badge/ci-passing-brightgreen.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Multiclass CNN for melanoma/skin lesion classification. Based on ISIC-like classes, with augmentation, imbalance handling, and reproducible training via config.

## Quickstart

```bash
pip install ".[dev]"
csc-train --config config/default.yaml
csc-eval  --model runs/*/model.keras
csc-predict --model runs/*/final_model.keras path/to/image1.jpg path/to/image2.jpg
```

## Repo layout
```
src/cnn_skin_cancer/   # package (train/eval/predict)
config/default.yaml    # hyperparameters, paths
assets/hero.png        # README & OpenGraph preview
```

## Notes
- Uses RMSprop (lr=1e-4, rho=0.9, eps=1e-8, decay=1e-6) as in your original notebook.
- Expects directory datasets: `data/train/<class>/*`, `data/val/<class>/*`.
=======
Convolutional neural network on images following augmentation and treatment of class imbalance on skin cancer data

# Multiclass classification model for Melanoma detection
> In this project, I have used convolutional neural network (CNN) to create a multiclass classification model that can identify the type of skin cancer by analyzing images. I have used data augmentation techniques and addressed class imbalances to obtain a better performance of the classifier.


## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)


## General Information
- Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.
- The dataset consists of 2357 images of malignant and benign oncological diseases, which were formed from the International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, with the exception of melanomas and moles, whose images are slightly dominant.
- The data set contains the following diseases:

* Actinic keratosis
* Basal cell carcinoma
* Dermatofibroma
* Melanoma
* Nevus
* Pigmented benign keratosis
* Seborrheic keratosis
* Squamous cell carcinoma
* Vascular lesion
- I have used a CNN model having the following architecture:

_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 rescaling (Rescaling)       (None, 180, 180, 3)       0         
                                                                 
 conv2d (Conv2D)             (None, 180, 180, 64)      4864      
                                                                 
 max_pooling2d (MaxPooling2D  (None, 90, 90, 64)       0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 88, 88, 128)       73856     
                                                                 
 conv2d_2 (Conv2D)           (None, 86, 86, 128)       147584    
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 43, 43, 128)      0         
 2D)                                                             
                                                                 
 conv2d_3 (Conv2D)           (None, 41, 41, 256)       295168    
                                                                 
 conv2d_4 (Conv2D)           (None, 39, 39, 256)       590080    
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 19, 19, 256)      0         
 2D)                                                             
                                                                 
 conv2d_5 (Conv2D)           (None, 17, 17, 512)       1180160   
                                                                 
 conv2d_6 (Conv2D)           (None, 15, 15, 512)       2359808   
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 7, 7, 512)        0         
 2D)                                                             
                                                                 
 conv2d_7 (Conv2D)           (None, 5, 5, 256)         1179904   
                                                                 
 conv2d_8 (Conv2D)           (None, 3, 3, 256)         590080    
                                                                 
 max_pooling2d_4 (MaxPooling  (None, 1, 1, 256)        0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 256)               0         
                                                                 
 dense (Dense)               (None, 256)               65792     
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dense_2 (Dense)             (None, 9)                 1161      
                                                                 
=================================================================
- I have used RMSprop optimizer and used learning rate 0.0001, rho=0.9, epsilon=1e-08, decay=1e-6.

## Conclusions
- Conclusion 1 – CNN without augmentation, dropout, and class imbalance correction resulted in 50% validation accuracy with a tendancy to overfit.
- Conclusion 2 – CNN with augmentation, after applying dropout layers reduced the tendency to overfit.
- Conclusion 3 – CNN with class imbalance correction improved accuracy and reduced overfitting. Final accuracy was 70% after addressing class imbalance using Augmentor.


## Technologies Used
- library - pathlib, glob, matplotlib, numpy, pandas, tensorflow, keras, Augmentor

## Acknowledgements
- This is an academic project, inspired by UpGrad
- This project was created to fulfil PGP AIML Requirement.


## Contact
Created by [@drkaushiksarkar] - feel free to contact me! www.drkaushiks.com
>>>>>>> e0eb089f3e7da80f9bbe9367d1f737631cf94879
