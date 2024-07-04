import torch
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch . nn . functional as F
import numpy as np

def grayscale_to_triple(input_tensor):
    
    # Ensure the input tensor has the expected shape
    if len(input_tensor.shape) != 4 or input_tensor.shape[1] != 1:
        raise ValueError("Input tensor should have shape (N, 1, H, W).")

    # Repeat the single channel 3 times along the channel dimension to create an RGB tensor
    rgb_tensor = torch.repeat_interleave(input_tensor, 3, dim=1)

    return rgb_tensor

class ModelCTrn18(nn.Module):
    def __init__(self):
        super(ModelCTrn18, self).__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.convolution2d = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.fc_maxpool = nn.AdaptiveMaxPool2d((1, 1))
            
    def forward(self, x):
        x = grayscale_to_triple(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x) 
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.convolution2d(x)
        x = self.fc_maxpool(x)
        x = torch.flatten(x, 1)
        
        return x
    
class ModelCTrn18opt(nn.Module):
    def __init__(self):
        super(ModelCTrn18opt, self).__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        #self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.convolution2d = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.fc_maxpool = nn.AdaptiveMaxPool2d((1, 1))
            
    def forward(self, x):
        x = grayscale_to_triple(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x) 
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.convolution2d(x)
        x = self.fc_maxpool(x)
        x = torch.flatten(x, 1)
        
        return x

class ModelCTrn18now(nn.Module):
    def __init__(self):
        super(ModelCTrn18now, self).__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.convolution2d = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.fc_maxpool = nn.AdaptiveMaxPool2d((1, 1))
            
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x) 
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.convolution2d(x)
        x = self.fc_maxpool(x)
        x = torch.flatten(x, 1)
        
        return x

class ModelCTdnn1(nn.Module):
    def __init__(self):
        super(ModelCTdnn1, self).__init__()
        self.lin1 = nn.Linear(512**2, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.lin2 = nn.Linear(512,1)
        
            
    def forward(self, x):
        x = torch . flatten (x , 1)
        x = self.lin1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x


class ModelCTdnn2(nn.Module):
    def __init__(self):
        super(ModelCTdnn2, self).__init__()
        self.lin1 = nn.Linear(512**2, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.lin2 = nn.Linear(512,256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.lin3 = nn.Linear(256,128)
        self.lin4 = nn.Linear(128,32)
        self.bn4 = nn.BatchNorm1d(32)
        self.lin5 = nn.Linear(32,1)

        
            
    def forward(self, x):
        x = torch . flatten (x , 1)
        x = self.lin1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.lin3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.lin4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.lin5(x)
        return x

class ModelCTcnn1(nn.Module):
    def __init__(self):
        super(ModelCTcnn1, self).__init__()
        self.conv1 = nn.Conv2d (1 , 4 , kernel_size =(7 , 7) , stride =(2 , 2) , padding =(3 , 3) ,
            bias = False )
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d (4 , 8 , kernel_size =(7 , 7) , stride =(2 , 2) , padding =(3 , 3) ,
            bias = False )
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d (8 , 16 , kernel_size =(5 ,5) , stride =(2 , 2) , padding =(2 , 2) ,
            bias = False )
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d (16 , 64 , kernel_size =(3 ,3) , stride =(1 , 1) , padding =(1 , 1) ,
            bias = False )
        self.bn4 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(1024 , 512)
        self.fc2 = nn.Linear(512 , 1)

        
            
    def forward(self, x):
        x = self.conv1( x )
        x = self.bn1( x )
        x = F.relu( x )
        x = F.max_pool2d(x ,(2 ,2) )
        x = self.conv2( x )
        x = self.bn2( x )
        x = F.relu( x )
        x = F.max_pool2d(x ,(2 ,2) )
        x = self.conv3( x )
        x = self.bn3( x )
        x = F.relu( x )
        x = F.max_pool2d(x ,(2 ,2) )
        x = self.conv4( x )
        x = self.bn4( x )
        x = F.relu( x )
        x = F.max_pool2d(x ,(2 ,2) )
        x = torch.flatten(x , 1)
        x = F.relu( self . fc1 ( x ) )
        x = self.fc2( x )
        return x
    
    
class ModelCTcnn2(nn.Module):
    def __init__(self):
        super(ModelCTcnn2, self).__init__()
        self.conv1 = nn.Conv2d (1 , 4 , kernel_size =(7 , 7) , stride =(2 , 2) , padding =(3 , 3) ,
            bias = False )
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d (4 , 8 , kernel_size =(7 , 7) , stride =(2 , 2) , padding =(3 , 3) ,
            bias = False )
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d (8 , 16 , kernel_size =(5 ,5) , stride =(2 , 2) , padding =(2 , 2) ,
            bias = False )
        self.bn3 = nn.BatchNorm2d(16)
        self.conv32 = nn.Conv2d (16 , 32 , kernel_size =(5 ,5) , stride =(2 , 2) , padding =(2 , 2) ,
            bias = False )
        self.bn32 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d (32 , 64 , kernel_size =(3 ,3) , stride =(1 , 1) , padding =(1 , 1) ,
            bias = False )
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d (64 , 128 , kernel_size =(3 ,3) , stride =(1 , 1) , padding =(1 , 1) ,
            bias = False )
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d (128 , 256 , kernel_size =(3 ,3) , stride =(1 , 1) , padding =(1 , 1) ,
            bias = False )
        self.bn6 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(2048 , 512)
        self.fc2 = nn.Linear(512 , 1)

        
            
    def forward(self, x):
        x = self.conv1( x )
        x = self.bn1( x )
        x = F.relu( x )
        x = F.max_pool2d(x ,(2 ,2) )
        x = self.conv2( x )
        x = self.bn2( x )
        x = F.relu( x )
        x = F.max_pool2d(x ,(2 ,2) )
        x = self.conv3( x )
        x = self.bn3( x )
        x = F.relu( x )
        x = self.conv32( x )
        x = self.bn32( x )
        x = F.relu( x )
        x = F.max_pool2d(x ,(2 ,2) )
        x = self.conv4( x )
        x = self.bn4( x )
        x = F.relu( x )
        x = self.conv5( x )
        x = self.bn5( x )
        x = F.relu( x )
        
        
        x = torch.flatten(x , 1)
        x = F.relu( self . fc1 ( x ) )
        x = self.fc2( x )
        return x