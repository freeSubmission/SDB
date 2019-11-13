"""
Code source: https://github.com/pytorch/vision
"""
from __future__ import absolute_import
from __future__ import division

__all__ = ['resnet']

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import resnet50, Bottleneck
import copy
import math
import random
from .pc import *


class BatchDrop(nn.Module):
    def __init__(self, h_ratio=0.3, w_ratio=1, Threshold=1):
        super(BatchDrop, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio
        self.it = 0
        self.Threshold = Threshold
        self.sx = None
        self.sy = None
    
    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(self.h_ratio * h)
            rw = round(self.w_ratio * w)
            if self.it % self.Threshold == 0:
                self.sx = random.randint(0, h-rh)
                self.sy = random.randint(0, w-rw)
            self.it += 1
            mask = x.new_ones(x.size())
            mask[:, :, self.sx:self.sx+rh, self.sy:self.sy+rw] = 0
            x = x * mask
        return x


class ResNet(nn.Module):

    def __init__(self, num_classes, fc_dims=None, loss=None, dropout_p=None,  **kwargs):
        super(ResNet, self).__init__()
        
        resnet_ = resnet50(pretrained=True)
        
        self.loss = loss
        
        self.layer0 = nn.Sequential(
            resnet_.conv1,
            resnet_.bn1,
            resnet_.relu,
            resnet_.maxpool)
        self.layer1 = resnet_.layer1
        self.layer2 = resnet_.layer2

        self.pc1 = PC_Module(512)

        self.layer3 = resnet_.layer3
 
        self.pc2 = PC_Module(1024)
        layer4 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        layer4.load_state_dict(resnet_.layer4.state_dict())
        
        self.layer40 = nn.Sequential(copy.deepcopy(layer4))
        self.layer41 = nn.Sequential(copy.deepcopy(layer4))

        self.pam_module1 = PAM_Module(2048)
        self.pam_module2 = PAM_Module(2048)

        
        self.batch_drop = BatchDrop()
        
        self.res_part1 = Bottleneck(2048, 512) 
        self.res_part2 = Bottleneck(2048, 512)  
                
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(2048)

        self.classifier1 = nn.Linear(2048, num_classes)
        self.classifier2 = nn.Linear(2048, num_classes)
              
        nn.init.constant_(self.bn1.weight, 1.0)
        nn.init.constant_(self.bn1.bias, 0.0)
        nn.init.constant_(self.bn2.weight, 1.0)
        nn.init.constant_(self.bn2.bias, 0.0)
        nn.init.normal_(self.classifier1.weight, 0, 0.01)
        if self.classifier1.bias is not None:
            nn.init.constant_(self.classifier1.bias, 0)
        nn.init.normal_(self.classifier2.weight, 0, 0.01)
        if self.classifier2.bias is not None:
            nn.init.constant_(self.classifier2.bias, 0)

    def featuremaps(self, x):
        if self.training:
           b = x.size(0)
           x1 = x[:b//2, :, :, :]
           x2 = x[b//2:, :, :, :]
           x2 = self.batch_drop(x2)
           x = torch.cat([x1, x2], 0)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pc1(x)
        x = self.layer3(x)
        x = self.pc2(x)
        if self.training:
           x_1 = x[:b//2,:,:,:]
           x_2 = x[b//2:,:,:,:]
           x_1 = self.layer40(x_1)
           x_2 = self.layer41(x_2)
        else:
           x_1 = self.layer40(x)
           x_2 = self.layer41(x)

        return x_1, x_2

    def forward(self, x):
        f1, f2 = self.featuremaps(x)

        f1 = self.res_part1(f1)
        f1 = self.pam_module1(f1)

        f2 = self.res_part2(f2)
        f2 = self.pam_module2(f2)
        
        v1 = self.global_avgpool(f1)
        v2 = self.global_maxpool(f2)

        v1 = v1.view(v1.size(0), -1)
        v2 = v2.view(v2.size(0), -1)

        fea = [v1, v2]
        
        v1 = self.bn1(v1)
        v2 = self.bn2(v2)
        

        if not self.training:
           v1 = F.normalize(v1, p=2, dim=1)
           v2 = F.normalize(v2, p=2, dim=1)
           return torch.cat([v1, v2], 1)

        y1 = self.classifier1(v1)
        y2 = self.classifier2(v2)

        if self.loss == 'softmax':
            return y1, y2
        elif self.loss == 'triplet':
            return y1, y2, fea
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

def resnet(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        fc_dims=None,
        loss=loss,
        dropout_p=None,
        **kwargs
    )
    return model

