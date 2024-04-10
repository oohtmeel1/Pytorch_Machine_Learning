# Custom data loaders for training validation testing
import subprocess
import sys
import os
import torchvision.transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch
subprocess.check_call([sys.executable, "-m", "pip",
                       "install", "-r",
                       "requirements.txt", '--quiet'])
print('Done!, proceeding momentarily')


assert os.path.exists(os.path.join(os.getcwd(), 'train'))
print('train is there')
assert os.path.exists(os.path.join(os.getcwd(), 'val'))
print('val is there')
assert os.path.exists(os.path.join(os.getcwd(), 'test'))
print('test is there')
print('importing dependencies please wait')
i = input("""creating training, validation,
testing datasets
would you like to proceed? (yes or no)""",)
if i == 'yes':
    pass
else:
    print('exiting now')
    quit()
   
   
# --------------------------------------------------
img_transforms = v2.Compose([v2.Resize((64, 64)),
                             v2.RandomHorizontalFlip(p=0.5),
                             v2.RandomPhotometricDistort(p=0.5),
                             v2.ToTensor()]
                            )
# --------------------------------------------------


class Loading_training(Dataset):
    def __init__(self):
        self.selected_dataset_dir = os.path.join(os.path.join(os.getcwd(),
                                                              'train'))
        self.all_filenames = os.listdir(self.selected_dataset_dir)
        self.all_labels = pd.read_csv(os.path.join(os.getcwd(),
                                                   'train_labels_final.csv'))
        self.label_meanings = self.all_labels.columns.values.tolist()

    def __len__(self):
        """Weird"""
        return len(self.all_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.selected_dataset_dir,
                                self.all_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = self.all_labels.iloc[idx, 1]
        label = torch.tensor(label)
        image = img_transforms(image)

        return image, label


class Loading_val(Dataset):

    def __init__(self):
        """Loading pet images"""

        self.selected_dataset_dir = os.path.join(os.path.join(os.getcwd(),
                                                              'val'))
        self.all_filenames = os.listdir(self.selected_dataset_dir)
        self.all_labels = pd.read_csv(os.path.join(os.getcwd(),
                                                   'val_labels_final.csv'))
        self.label_meanings = self.all_labels.columns.values.tolist()

    def __len__(self):
        """Weird"""
        return len(self.all_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.selected_dataset_dir,
                                self.all_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = self.all_labels.iloc[idx, 1]
        label = torch.tensor(label)
        image = img_transforms(image)
        return image, label


class Loading_test(Dataset):
    def __init__(self):

        """Loading pet images"""
        self.selected_dataset_dir = os.path.join(os.path.join(os.getcwd(),
                                                              'test'))
        self.all_filenames = os.listdir(self.selected_dataset_dir)
        self.all_labels = pd.read_csv(os.path.join(os.getcwd(),
                                                   'test_labels_final.csv'))
        self.label_meanings = self.all_labels.columns.values.tolist()

    def __len__(self):
        """Weird"""
        return len(self.all_filenames)

    def __getitem__(self, idx):

        img_path = os.path.join(self.selected_dataset_dir,
                                self.all_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = self.all_labels.iloc[idx, 1]
        label = torch.tensor(label)
        image = img_transforms(image)
        return image, label


# --------------------------------------------------
print('loading training data')
training1 = Loading_training()
print('loading validation data')
val1 = Loading_val()
print('loading testing data')
test1 = Loading_test()
# --------------------------------------------------
