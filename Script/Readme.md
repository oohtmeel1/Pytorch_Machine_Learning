So what this script is intended to do:


The first file: \
`loading_data_c_.py` <-- Creates labels from existing images of dogs and cats specifically and saves them to csv files in the current working directory. 
Please be careful and dont load random junk.
Takes a few arguments:
`file` is the type of file, either train test or validation. 
`dir` is the directory name where you want the labels to go.
It does ask if you want to overwrite existing labels. And does not allow for specification of file names.

The second file:
`dataloader_classes.py` <-- Loads and transforms all images into tensors, and feeds the labels in. 
