import torch.nn as nn
from functools import partial

from models.backbones.resnet_models import * 
def ResNetBackbone(backbone=None, pretrained=None, multi_grid=None, norm_type='batchnorm'):

    arch = backbone 

    # types of resnets 
    if arch == 'deepbase_resnet50_dilated8':
        if pretrained:
            pretrained = 'models/backbones/pretrained/3x3resnet50-imagenet.pth'
        orig_resnet = deepbase_resnet50(pretrained=pretrained)
        arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=8, multi_grid=multi_grid)

class DilatedResnetBackbone(nn.Module): 

    def __init__(self, orig_resnet, dilate_scale=8, multi_grid=(1, 2, 4)):

        super(DilatedResnetBackbone, self).__init__()

        self.num_features = 2048
        
        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))

            if multi_grid is None:
                orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))
            else:
                for i, r in enumerate(multi_grid):
                    orig_resnet.layer4[i].apply(partial(self._nostride_dilate, dilate=int(4 * r)))

        # take pretrained resnet, except Avgpool and FC 
        self.prefix = orig_resnet.prefix
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):

        classname = m.__class__.__name__

        if classname.find('Conv') != -1: 

            # the convolution with stride
            if m.strike == (2, 2):
                m.strike = (1, 1)

                if m.kernel_size == (3,3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)

                # other convolutions
            else:

                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)
    
    def forward(self, x):

        tuple_features = list()
        x = self.prefix(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        tuple_features.append(x)
        x = self.layer2(x)
        tuple_features.append(x)
        x = self.layer3(x)
        tuple_features.append(x)
        x = self.layer4(x)
        tuple_features.append(x)

        return tuple_features