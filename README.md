# Machine Learning (CS-GY6923) Final Project

This repository and project was developed for the graduate level machine learning class CS-GY-6923 at NYU. The class professor is Linda Sellie. 

The author of the project is [Justin Snider](https://github.com/aobject/). 

## Introduction

In this project we implements three extensions to the basic neural network machine learning strategies introduce in the class. For each extension we demonstrate how the strategy can be implemented using the libraries [scikit-learn](https://scikit-learn.org/stable/) and [PyTorch](https://pytorch.org/). Then, we will implement the strategy ourselves using only [NumPy](https://numpy.org/). 

The first extension we develop is [convolution layers](##Extension-01-//-Neural-Network-with-Convolution). Second, we build on CNN to introduce the [use of pooling](##Extension-02-//-Pooling). For the final extension we introduce the use of [skip links](##Extension-03-//-Neural-Networks-with-Skip-Links). 

We use two datasets to evaluate the performance of our code. First, we use the [SciKit Hand Written Digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) used in class. This dataset has 10 classes, which are the digits 0 through 9. There are 1,797 sample. Each sample is in the format of an 8 x 8 pixel grid. The original source of the data used by SciKit is the [UCI Machine Learnign Respository](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits). Here we have an example of a zero from the dataset. 

![mnist number 0](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/images/number.png)



The second dataset is the more challenging [CIFAR-10 ](https://www.cs.toronto.edu/~kriz/cifar.html). We again have 10 classes. There are 60,000 total images, with 6,000 images per class. However, for the sake of efficiency we use just 1,000 images randomly selected. The images are formated as 32 x 32 pixels with 3 color channels. Here we have an example of 10 images from each of the 10 classes. 

![cifar image](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/images/cifar.png)





## Required Setup

1. Just click on the <a href="https://colab.research.google.com/github/aobject/NYU-ML-Project/blob/master/Extension-1/extended_implementation_using_numpy.ipynb" style="text-decoration: none;font-family: Roboto, sans-serif;color: white;font-weight: bold;font-size: 16px;padding:8px;margin-top:4px;8px;margin-bottom:4px;background:#00B4FF;">Open in Colab</a> links provided in this guide and you are ready to go! 

   

   Inside each of the live Google Colab notebooks code is provided to allow you to mount Your Google Drive and load the data used to a 'temp-data' inside the root 'My Drive' folder. The data downloaded included is a limited and compressed portion of the MNIST and CIFAR-10 datasets. When you are done running these notebooks you can simply delete the 'temp-data' folder from Google Drive to recover the space. 



## Benchmarks Neural Network Without Extension

For consistency sake we use the same training and test sets for all the tests. In addition, we visualize the first 10 epochs of all tests in the same manor. You will find for all the successful code sets included a visualization of the loss, the accuracy, and the accuracy against the benchmark. 

Using the interactive graphs in the live Google Colabs notebook you can get all the specific value by hovering your mouse over the point you want to investigate. 

The Neural Network code provided in class uses the [scikit-learn](https://scikit-learn.org/stable/) and  [NumPy](https://numpy.org/) libraries to learn and predict the hand written digit represented in an 8 by 8 pixel image from the [SciKit Hand Written Digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)  dataset. For completeness we have included the performance here.

![acc](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-01/pytorch/00-mnist-baseline-hw-loss.png) ![acc](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-01/pytorch/00-mnist-baseline-hw-acc.png)

To establish a better MNIST baseline we created a PyTorch Neural network using the same 8 x 8 pixel dataset. Here is the custom PyTorch dataset class code:

```python
class MNISTDataset(Dataset):
    def __init__(self, data, label):
        self.data = data.reshape((-1,8,8,1))
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = self.data[item].transpose((2, 0, 1))
        image = torch.from_numpy(image)
        target = self.label[item]
        target = torch.from_numpy(target)
        return (image, target)
```

Our PyTorch Neural Network class uses the same neural network structure with 64 input nodes, 30 hidden nodes, and 10 output nodes. In addition, we use the same sigmoid activation as the original homework assignment code. Here is the Neural Network class:

```python
class MNIST(nn.Module):
    # Our batch shape for input x is (1, 8, 8)
    def __init__(self):
        super(MNIST, self).__init__()
        self.fc1 = nn.Linear(64, 30)
        self.fc2 = nn.Linear(30, 10)

    def forward(self, x):
        x = x.view(-1, 8 * 8)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
```

We optimized the hyper-parameters and used the following:

| Hyper-parameter | Value |
| :-------------- | ----- |
| batch size      | 1     |
| learning rate   | 0.01  |
| epochs          | 10    |

We have the following baseline performance for our naive neural network on the MNIST dataset. 

![acc](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-01/pytorch/01-mnist-baseline-acc.png) ![acc](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-01/pytorch/01-mnist-baseline-loss.png)

The set up is very similar for the CIFAR baseline naive neural network. We changed the network architecture to accommodate the larger input image size and additional channels. The CIFAR-10 dataset is much more challenging than the MNIST dataset and we see a decrease in performance to illustrate that difficulty. 

![acc](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-01/pytorch/03-cifar-baseline-loss.png) ![acc](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-01/pytorch/03-cifar-baseline-acc.png)



## Extension 01 // Neural Network with Convolution

Description here.



### Convolutional Neural Network Implemented in NumPy

<a href="https://colab.research.google.com/github/aobject/NYU-ML-Project/blob/master/Extension-1/extended_implementation_using_numpy.ipynb" style="text-decoration: none;font-family: Roboto, sans-serif;color: white;font-weight: bold;font-size: 16px;padding:8px;margin-top:4px;8px;margin-bottom:4px;background:#00B4FF;">Open in Colab</a>



Description here.



![loss](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-01/numpy/01-mnist-numpy-sigmoid-loss.png)  ![acc](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-01/numpy/01-mnist-numpy-sigmoid-acc.png)



![acc with relu](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-01/numpy/02-mnist-numpy-relu-loss.png) ![acc with relu](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-01/numpy/02-mnist-numpy-relu-acc.png)





![acc with relu](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-01/numpy/03-cifar-numpy-sigmoid-loss.png) ![acc with relu](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-01/numpy/03-cifar-numpy-sigmoid-acc.png)



![acc with relu](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-01/numpy/04-cifar-numpy-relu-loss.png) ![acc with relu](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-01/numpy/04-cifar-numpy-relu-acc.png)



####  

![acc with relu](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-01/numpy/05-synopsis-mnist-acc.png) ![acc with relu](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-01/numpy/05-synopsis-cifar-acc.png)







### Convolutional Neural Network Implemented in PyTorch

<a href="https://colab.research.google.com/github/aobject/NYU-ML-Project/blob/master/Extension-1/scikit_pytorch_implementation.ipynb" style="text-decoration: none;font-family: Roboto, sans-serif;color: white;font-weight: bold;font-size: 16px;padding:8px;margin-top:4px;8px;margin-bottom:4px;background:#00B4FF;">Open in Colab</a>

description here. 

####  

![image](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-01/pytorch/02-mnist-pytorch-loss.png) ![image](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-01/pytorch/02-mnist-pytorch-acc.png)



![image](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-01/pytorch/04-cifar-pytorch-loss.png) ![image](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-01/pytorch/04-cifar-pytorch-acc.png)









![image](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-01/pytorch/05-synopsis-mnist.png) ![image](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-01/pytorch/05-synopsis-cifar.png)



## Extension 02 // Pooling

Description here. 



### CNN with Pooling in NumPy

<a href="https://colab.research.google.com/github/aobject/NYU-ML-Project/blob/master/Extension-2/extended_implementation_using_numpy.ipynb" style="text-decoration: none;font-family: Roboto, sans-serif;color: white;font-weight: bold;font-size: 16px;padding:8px;margin-top:4px;8px;margin-bottom:4px;background:#00B4FF;">Open in Colab</a>

Description here. 

![image](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-02/numpy/01-mnist-numpy-loss.png) ![image](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-02/numpy/01-mnist-numpy-acc.png)

![image](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-02/numpy/02-cifar-numpy-loss.png) ![image](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-02/numpy/02-cifar-numpy-acc.png)



![image](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-02/numpy/03-synopsis-mnist-acc.png) ![image](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-02/numpy/03-synopsis-cifar-acc.png)





### CNN with Pooling in PyTorch

<a href="https://colab.research.google.com/github/aobject/NYU-ML-Project/blob/master/Extension-2/scikit_pytorch_implementation.ipynb" style="text-decoration: none;font-family: Roboto, sans-serif;color: white;font-weight: bold;font-size: 16px;padding:8px;margin-top:4px;8px;margin-bottom:4px;background:#00B4FF;">Open in Colab</a>

Description here. 

![image](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-02/pytorch/01-mnist-pytorch-loss.png) ![image](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-02/pytorch/01-mnist-pytorch-acc.png)



![image](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-02/pytorch/02-cifar-pytorch-loss.png) ![image](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-02/pytorch/02-cifar-pytorch-acc.png)









![image](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-02/pytorch/03-synopsis-mnist-acc.png) ![image](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-02/pytorch/03-synopsis-cifar-acc.png)



## Extension 03 // Neural Networks with Skip Links 

Description here.



### CNN with Skip Link Extension in NumPy

<a href="https://colab.research.google.com/github/aobject/NYU-ML-Project/blob/master/Extension-3/extended_implementation_using_numpy.ipynb" style="text-decoration: none;font-family: Roboto, sans-serif;color: white;font-weight: bold;font-size: 16px;padding:8px;margin-top:4px;8px;margin-bottom:4px;background:#00B4FF;">Open in Colab</a>

Description here. 



![image](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-03/numpy/01-mnist-loss.png) ![image](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-03/numpy/01-mnist-acc.png)



![image](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-03/numpy/02-cifar-loss.png) ![image](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-03/numpy/02-cifar-acc.png)





![image](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-03/numpy/03-synopsis-mnist-acc.png) ![image](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-03/numpy/03-synopsis-cifar-acc.png)





### CNN with Skip Link Extension in PyTorch

<a href="https://colab.research.google.com/github/aobject/NYU-ML-Project/blob/master/Extension-3/scikit_pytorch_implementation.ipynb" style="text-decoration: none;font-family: Roboto, sans-serif;color: white;font-weight: bold;font-size: 16px;padding:8px;margin-top:4px;8px;margin-bottom:4px;background:#00B4FF;">Open in Colab</a>

Description here. 

![image](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-03/pytorch/01-mnist-pytorch-loss.png) ![image](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-03/pytorch/01-mnist-pytorch-acc.png)

![image](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-03/pytorch/02-cifar-loss.png) ![image](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-03/pytorch/02-cifar-acc.png)



![image](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-03/pytorch/03-synopsis-mnist-acc.png) ![image](https://raw.githubusercontent.com/aobject/public-nyu-ml/master/ML-Project/results/extension-03/pytorch/03-synopsis-cifar-acc.png)



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

   * Dimensionality: 32 x 32 x 3 color images



## Bibliography

Research Papers and Online Resources: 

1. [PyTorch](https://pytorch.org/)
2. [Scikit-learn](https://scikit-learn.org/)
3. [NumPy](https://numpy.org/)
4. [Performance Benchmarks Organized by Dataset](https://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html)
5. CNN papers and links here...
6. Pooling and ReLu papers and links here...
7. Skip Link papers and links here... 
  <!--stackedit_data:
  eyJoaXN0b3J5IjpbLTU3OTMxMjAxNCwtMTA0NzIwOTQwM119
  -->