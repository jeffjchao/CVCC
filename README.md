# Clothing Image Classification Using Computer Vision 
While the MNIST dataset is often seen as the “hello world” into image 
recognition, it quickly runs into issues when attempting to use it for advanced machine 
learning due to the extremely high accuracy that can be reached with classifier tuning or 
using advanced neural network techniques. In response to this ceiling of usability, the 
Fashion-MNIST dataset was developed.

The Fashion-MNIST dataset was designed to serve as a direct drop-in 
replacement for the original MNIST dataset; it shares the same image size and structure 
of the training and testing splits from its predecessor. The training set of 60,000 
grayscale images with dimensions of 28x28 pixels is labeled into 10 different types of 
clothing and an additional testing set of 10,000.

## Obtaining The Data
There are many different options to acquiring the Fashion-MNIST dataset. 
Zalando, the company who developed the dataset, has the data hosted on Github to 
encourage ‘serious machine learning researchers’ to download and use it. Furthermore, 
the Fashion-MNIST dataset is hosted on many machine learning libraries like Kaggle, 
Pytorch, Keras, and Tensorflow. Because Tensorflow is one of the tools used to create 
evaluate the classifiers, the Fashion-MNIST dataset was imported from there.

## Exploratory Data Analysis
Taking a dive into summary statistics about the Fashion-MNIST dataset, there is 
not much benefit in quantitatively analyzing each attribute/pixel. However, some brief 
insights can be gleamed from the distribution of the labeling data; there are the same 
amount of each labeled clothing, resulting in 6000 of each in the training data and 1000 
of each in the testing data. This allows for proceeding with the classifier evaluation 
without worrying about imbalanced classification problems since most machine learning 
algorithms are designed for use on datasets with an equal distribution of classes.

## Models:
### Naive Bayes
### k-Nearest Neighbors
### Random Forest
### Support Vector Machines
### Multilayer Perceptron
### Keras Baseline
### Convolutional Neural Network
### LeNet-5 Imitation
### 19-Layer CNN
