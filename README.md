# Machine Learning (CS-GY6923) Final Project

This repository and project was developed for the CS-GY-6923 Machine Learning class at NYU. The class professor is Linda Sellie. 

The author of the project is [Justin Snider](https://github.com/aobject/). 









## Introduction

This project implements three extensions to machine learning strategies covered in the class. For each extension we demonstrate how the strategy can be implemented using the [scikit-learn](https://scikit-learn.org/stable/) and [PyTorch](https://pytorch.org/). Then, we will implement the strategy using only Numpy [NumPy](https://numpy.org/). 

## Required Setup

1. Open the notebook you would like to run in Google Colab via the links provided in this guide and you are ready to go! 

   In side each notebook code is provided to allow you to mount Your Google Drive and load the data used to a 'temp-data' inside the root 'My Drive' folder. The data downloaded included is a limited and compressed portion of the MNIST and CIFAR-10 datasets. When you are done running these notebooks you can simply delete the 'temp-data' folder from Google Drive to recover the space. 




## Extension 01 // Neural Network CNN

Description here. 

### Convolutional Neural Network Implemented in NumPy

[![Generic badge](https://img.shields.io/badge/CNN NumPy-Open in Colab-black.svg)](https://colab.research.google.com/github/aobject/NYU-ML-Project/blob/master/Extension-1/scikit_pytorch_implementation.ipynb)

[![Generic badge](https://img.shields.io/badge/CNN NumPy-View in GitHub-black.svg)](https://github.com/aobject/NYU-ML-Project/blob/master/Extension-1/extended_implementation_using_numpy.ipynb)



[![forthebadge test](http://ForTheBadge.com/images/badges/no-ragrets.svg)](https://www.python.org/)



### Convolutional Neural Network Implemented in PyTorch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aobject/NYU-ML-Project/blob/master/Extension-1/scikit_pytorch_implementation.ipynb)



### 





## Extension 02 // Pooling and ReLu

```
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb)
```

Description here. 

## Extension 03 // Neural Network Deep Neural Network Skip Link Blocks

Description here.



## Directories

Extension-1/		

* The directory containing the Baseline Neural Network without CNN and a Convolutional Neural Network extension. The extension is implemented in PyTorch and separately in NumPy. There are unique implementations of both for the two datasets MNIST and CIFAR-10. In addition, we explore the benefits that can be gained using Leaky ReLu over a Sigmoid activation. 

Extension-2/

* The directory contains a baseline neural network without extension. In addition, there are Convolutional Neural Networks that have been extended with Pooling layers. Pooling has been implemented using NumPy only and with PyTorch. 

Extension-3/

* The directory containing a baseline neural network without extension. In addition, there are convolutional neural networks that have been extended with the skip link strategy used by deep neural networks. Skip link has been implemented using NumPy only and with PyTorch. 



## Datasets

1. [SciKit Hand Written Digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
	* Classes: 10 total including the digits 0 - 9
	* Total Samples: 1,797
	* Dimensionality: 64 including all values from an 8x8 grid.
	* Values: Integers 0 - 16. 
	* [Original Source at UCI Machine Learnign Respository](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits)
2. [CIFAR-10 ](https://www.cs.toronto.edu/~kriz/cifar.html)
	* Classes: 10
	* Total Images: 60000, with 6000 images per class
	* Training Images: 50000 training images
	* Test Images: 10000
	* Dimensionality: 32x32 colour images

## Bibliography

Research Papers and Online Resources: 

1. [PyTorch](https://pytorch.org/)
2. [Scikit-learn](https://scikit-learn.org/)
3. Numpy [NumPy](https://numpy.org/)
4. [Performance Benchmarks Organized by Dataset](https://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html)
5. CNN papers and links here...
6. Pooling and ReLu papers and links here...
7. Skip Link papers and links here... 
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTU3OTMxMjAxNCwtMTA0NzIwOTQwM119
-->