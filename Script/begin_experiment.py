#!/usr/bin/env python3
"""Lo"""
import argparse
from typing import NamedTuple
import os
import pandas as pd
import torch
from model_architecture import Modelo
from dataloader_classes import Loading_training
from dataloader_classes import Loading_val
from dataloader_classes import Loading_test
from torch.utils.data import DataLoader


class Args(NamedTuple):
    """ Command-line arguments """
    begin: str

# --------------------------------------------------

    
### This portion just inputs the information 
def main():
    l_ = input('please specify the type of device to use: (cuda or cpu)',)
    if l_ == 'cuda':
        device = torch.device('cuda')
        print(device)
    else:
        device = torch.device('cpu')
        print(device)
    l_ = input('going to load model to device)(y/n)',)
    ##This stores the model to the device after calling the correct function
    if l_ == 'y':
        model = Modelo()
        model.to(device)
    else:
        return
    ## This loads the training, validation and testing data sets
    training = Loading_training()
    val = Loading_val()
    test = Loading_test()
    print('loading data')
    
    def dataloading(loader, batch_size, shuffle_):
        x = DataLoader(loader, batch_size = batch_size, shuffle = shuffle_)
        return x
    l_ = ['train','val','test']
    for i in l_:
        print(f'dataloader_{i}, creating dataloaders')
        exec(f'dataloader_{i} = dataloading("{i}",16,True)')
        
    def trainer_time(loss_fn,trainer):
        
        

    
          
    
        
    
# --------------------------------------------------

if __name__ == '__main__':
    main()   