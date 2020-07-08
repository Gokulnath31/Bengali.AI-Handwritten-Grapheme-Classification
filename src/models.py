import pretrainedmodels
import torch.nn as nn
from torch.nn import functional as F
 
 
class ResNet34(nn.Module):   
    def __init__(self ,pretrained):
        super(ResNet34,self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained=None)

        # To replace the last layer of the model with these
        self.l0 = nn.Linear(512,168) # 168 grapheme_root
        self.l1 = nn.Linear(512,11) # 11 vowel_diacritic
        self.l2 = nn.Linear(512,7) # 7 consonant_diacritic

    def forward(self,x):
#         print(x.shape)
        batch_size ,_,_,_ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size, -1)
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)
        return l0, l1, l2 # grapheme_root, vowel_diacritic, consonant_diacritic