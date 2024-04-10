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
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine
from ignite.metrics import Accuracy
from ignite.metrics import Precision, Recall
from ignite.handlers import ModelCheckpoint
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator  
from ignite.metrics import Accuracy, Loss
from ignite.handlers import Timer, BasicTimeProfiler, HandlersTimeProfiler    
from ignite.engine import Events
import warnings
warnings.filterwarnings('ignore')

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
    train = Loading_training()
    val = Loading_val()
    test = Loading_test()
    print('loading data')
    
    def dataloading(loader, batch_size, shuffle_):
        x = DataLoader(loader, batch_size = batch_size, shuffle = shuffle_)
        return x
    
    train_length= pd.read_csv('training_labels_final.csv')
    q = len(train_length)//100
    print(q)
    train_dataloader = dataloading(train, q, True)
    val_dataloader = dataloading(val, 16, True)
    
        
        
    loss_fn_ = input('Please enter a loss function ending in (), else nn.BCEWithLogitsLoss() will be used',)
    optimizer = input('Please enter an optimizer function, else torch.optim.AdamW will be used,',)
    x_ = input('Please enter a learning rate speed,else 0.00005 will be used',)
    from torch import nn
    if loss_fn_ == "":
        loss_fn_ = nn.BCEWithLogitsLoss()
    else:
        loss_fn_ = f"{loss_fn_}"
    if x_ == "":
        x_ = 0.00005
    else:
        x_ = x_
    if optimizer == "":
        optimizer = torch.optim.AdamW(model.parameters(), lr = x_)
    else:
        optimizer = (f"torch.optim{optimizer}(model.parameters(), lr = {x_})")
    print(loss_fn_)
    print(optimizer) 
     
    def update_model(engine, batch):
        model.train()
        data,label = batch
        optimizer.zero_grad()
        data=data.to(device)
        label=label.to(device)
        #print(label.get_device())
        #print(data.get_device())
        outputs,_=(model(data),label)
        outputs=outputs.squeeze()
        loss = loss_fn_(outputs, label.float())
        loss.backward()
        optimizer.step()
        return loss.item()

    trainer = Engine(update_model)
 

    val_metrics = {
        "accuracy": Accuracy(),
        "loss": Loss(loss_fn_)
    }


    def validation_step(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = batch
            x=x.to(device)
            y=y.to(device)
            outputs,_=(model(x),y)
            outputs = outputs.squeeze()
            outputs=torch.sigmoid(outputs)
            outputs=outputs.round()
            y=y.cpu().detach()

        return outputs.float(), y.float()
        
        
    evaluator = Engine(validation_step)




    precision = Precision()


    Accuracy().attach(evaluator, "accuracy")
    Precision().attach(evaluator,'precision')
    Recall(average='weighted').attach(evaluator,'recall')


        

    @trainer.on(Events.ITERATION_COMPLETED(every=50))
    def log_training(engine):
        batch_loss = engine.state.output
        lr = optimizer.param_groups[0]['lr']
        e = engine.state.epoch
        n = engine.state.max_epochs
        i = engine.state.iteration
        print("Epoch {}/{} : {} - batch loss: {}, lr: {}".format(e, n, i, batch_loss, lr))
        
        
        
        
        


    validate_every = 5


    @trainer.on(Events.EPOCH_COMPLETED(every=validate_every))
    def run_validation():
        evaluator.run(val_dataloader)
        
    @trainer.on(Events.EPOCH_COMPLETED(every=validate_every))
    def log_validation():
        ugh=[]
        metrics = evaluator.state.metrics
        print(f"Epoch: {trainer.state.epoch},  Accuracy: {metrics['accuracy']},  Precision: {metrics['precision']}, recall: {metrics['recall']}")


    def score_function(engine):
        return engine.state.metrics["accuracy"]


    model_checkpoint = ModelCheckpoint(
        "checkpoint",
        n_saved=25,
        filename_prefix="best",
        score_function=score_function,
        score_name="accuracy",
        global_step_transform=global_step_from_engine(trainer),
    )
    
    evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

    tb_logger = TensorboardLogger(log_dir="tb-logger")

    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=100),
        tag="training",
        output_transform=lambda loss: {"batch_loss": loss},
    )

    for tag, evaluator in [("training", trainer), ("validation", evaluator)]:
        tb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names="all",
            global_step_transform=global_step_from_engine(trainer),
        )

    trainer.run(train_dataloader, max_epochs=5)
        
    
# --------------------------------------------------

if __name__ == '__main__':
    main()   