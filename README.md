# Pytorch_Machine_Learning
Machine Learning project using Python and Pytorch for binary Classification of images.


This notebook shows the workflow for building, training, and testing a custom model class using Pytorch for Machine learning. The specific purpose is Binary Classification of images. This notebook covers up to the end of the training portion. Training is implemented using Pytorch Ignite.


Project Overview: Images are cleaned, information is visualized and analysis is run.

The images are first cleaned, duplicate checking is performed using the ImageHash package. \
A transformer is initialized. Which performs various types of data augmentation.
Pytorch Custom Data set classes are then implemented.
Pytorch Ignite is used for model training and to create model checkpoints.
The model implemented is a custom neural network classifier.
The model is tested against a subset of the original data.
The model is then tested on completely unseen data.
