import torch
from torch import nn

class Modelo(nn.Module):
    
    def __init__(self):
        super(Modelo, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            #torch.nn.Dropout2d(p=0.2),
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=1,bias=False),
            nn.BatchNorm2d(32, affine=False),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.LeakyReLU(inplace=True,negative_slope=0.02))
        self.main1 = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1),
        torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1),
        torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1),
        torch.nn.Dropout2d(p=0.2),
        torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=1),
        torch.nn.LeakyReLU(inplace=True,negative_slope=0.02))
        self.main2 = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1),
        torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1),
        torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1),
        torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=1),
        #torch.nn.MaxPool2d(2, 2),
        torch.nn.LeakyReLU(inplace=True,negative_slope=0.02))
        self.main3 = torch.nn.Sequential(
        torch.nn.Dropout2d(p=0.2),
        torch.nn.Conv2d(in_channels=8, out_channels=2, kernel_size=(3, 3), padding=1),
        torch.nn.MaxPool2d(2, 2),
        torch.nn.LeakyReLU(inplace=True,negative_slope=0.1))
        #self.fc1 = nn.Linear(in_features=8192,out_features=4096)
        #self.fc2 = nn.Linear(in_features=4096,out_features=2048)
        #self.fc3 = nn.Linear(in_features=2048,out_features=1024)
        self.fc4 = nn.Linear(in_features=512,out_features=256)
        self.fc5 = nn.Linear(in_features=256,out_features=128)
        self.fc6 = nn.Linear(in_features=128,out_features=1)

    def forward(self, x):
        x = self.main(x)
        x = self.main1(x)
        x = self.main2(x)
        x = self.main3(x)
        x = torch.flatten(x,start_dim=1,end_dim=-1)
        #x = self.fc1(x)
        #x = self.fc2(x)
        #x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        return x



