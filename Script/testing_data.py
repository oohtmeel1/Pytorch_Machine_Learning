

from torcheval.metrics.functional import binary_accuracy
from torch.utils.data import DataLoader
from torcheval.metrics.functional.classification import binary_recall
from torcheval.metrics.functional.aggregation.auc import auc
from torcheval.metrics.functional import binary_precision
from torcheval.metrics.functional import binary_accuracy
from torcheval.metrics.functional import binary_f1_score
from dataloader_classes import Loading_test
import os
import torch
from model_architecture import Modelo
from sklearn.metrics import roc_curve
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(0)
accuracy10=[]
roc_things=[]
model=Modelo()

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
	quit()
test = Loading_test()
test_dataloader = DataLoader(test, batch_size = 80, shuffle = True)

new_dir = os.path.join(os.getcwd(),'checkpoint')
for j in os.listdir(new_dir):
    PATH=os.path.join(new_dir,j)
    model.load_state_dict(torch.load(PATH))
    for i, batch in enumerate(test_dataloader):
        model.eval()
        data,label = batch[0].float(),batch[1]
        data = data.to(device)
        label = label.to(device)
        y_logits,label = (model(data),label)
        test_pred = y_logits.squeeze(-1)
        test_pred = torch.sigmoid(test_pred)
        test_pred = test_pred.cpu().detach()
        test_pred = test_pred.round()
        test_pred1 = test_pred.cpu().detach()
        label = label.cpu().detach()
        fpr, tpr, threshold = roc_curve(label.float(), test_pred1)
        
        bina_acc = binary_accuracy(test_pred, label, threshold=0.7)
        bin_acc = bina_acc.detach().numpy()
        
        f1_score1 = binary_f1_score(test_pred, label, threshold=0.61)
        f1_score = f1_score1.detach().numpy()
        
        bin_rec1 = binary_recall(test_pred, label, threshold=0.61)
        bin_rec = bin_rec1.detach().numpy()
        
        bin_prec1 = binary_precision(test_pred, label, threshold=0.61)
        bin_prec = bin_prec1.detach().numpy()
        
        accuracy10.append([bin_acc, f1_score,bin_rec,
                           bin_prec, j, test_pred1.numpy(),
                           label.numpy()])
        roc_things.append([fpr, tpr, threshold])
        
        ahh1=pd.DataFrame(accuracy10,columns = ['Accuracy','F1', 'binary_recall', 
                                     'binary_precision', 'model_info',
                                     'y_pred','y_true'])

        
        print(f" Accuracy: {ahh1['Accuracy']}, F1: {ahh1['F1']}, Precision: {ahh1['binary_precision']}, recall: {ahh1['binary_recall']}")