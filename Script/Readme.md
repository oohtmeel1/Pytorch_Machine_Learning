So what this set of scripts is intended to do:
Read current working directory for train, val, test files, make labels, and load pytorch model to perform analysis.

The first file: \
`loading_data_c_.py` <-- Creates labels from existing images of dogs and cats specifically and saves them to csv files in the current working directory. 
Please be careful and don't load random junk.
Takes a few arguments:
`file` is the type of file, either train test or validation. 
`dir` is the directory name where you want the labels to go.
It does ask if you want to overwrite existing labels. And does not allow for specification of file names.

The second file: \
`dataloader_classes.py` <-- Loads and transforms all images into tensors, matching the labels, loads dependencies.
It first checks to see if the folders, train val and test are present. And if they are, it proceeds.

The third file: 

`model_architecture.py` loads the Pytorch CNN.

The Fourth file:

`run_experiment.py` Runs the model, training it on data in the train directory and validating it on the val directory.


