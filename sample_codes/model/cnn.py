import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

from utils.options import args



device = torch.device(f"cuda:{args.gpus[0]}")



class CNN(nn.Module):

    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        
     
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2, bias=False)
        
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2, bias=False)
        self.maxpool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout2d(0.3)
        self.dropout3 = nn.Dropout2d(0.5)
        
        self.fc1 = nn.Linear(64, 256)
        
        self.fc2 = nn.Linear(256, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)
        
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = self.maxpool2(x)
        x = self.dropout2(x)
        
        x = self.avgpool(x)
        feature = torch.flatten(x, 1)
        x = self.fc1(feature)
        x = self.relu(x)

        x = self.dropout3(x)
        x = self.fc2(x)
        
        return x

