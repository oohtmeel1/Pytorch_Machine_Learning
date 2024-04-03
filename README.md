# Pytorch_Machine_Learning
Machine Learning project using Python and Pytorch. binary Classification of images.


This notebook shows the workflow for building, training, and testing a custom model class using Pytorch for Machine learning. The specific purpose is Binary Classification of images. This notebook covers up to the end of the training portion. Training is implemented using Pytorch Ignite.


Project Overview: The code in the cells of this notebook performs the following: The images are first cleaned, duplicate checking is performed using the ImageHash package. The data is visualized in various ways for inspection and inference. The image files contained in the train/test/val folders have label files created which are stored in the main working directory. The resulting file names will be: [“training_labels_final.csv”, “val_labels_final.csv”, “test_labels_final.csv”] A transformer is initialized. Which performs various types of data augmentation. ## Transformer img_transforms = v2.Compose([v2.Resize((64,64)),  Resizes the images. v2.RandomHorizontalFlip(p=0.5),  Flips the images. v2.RandomPhotometricDistort(p=0.5),  Distorts the images. v2.ToTensor()])  Turns images into tensors.
