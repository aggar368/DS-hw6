import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

from utils.options import args



device = torch.device(f"cuda:{args.gpus[0]}")



class CNN(nn.Module):

    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        
     
        self.conv1 = nn.Conv2d(3, 20, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))
        
        x = F.max_pool2d(x, 2)

        x = self.dropout1(x)

        feature = torch.flatten(x, 1)
        print(feature.shape)
        x = F.relu(self.fc1(feature))

        x = self.dropout2(x)

        x = self.fc2(x)
        

        return x

