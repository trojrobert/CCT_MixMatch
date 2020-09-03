import os
from models.backbones.resnet_backbone import ResNetBackbone

import torch.nn as nn

resnet50 = { "path": "models/backbones/pretrained/3x3resnet50-imagenet.pth",}

class Encoder(nn.Module):

    def __init__(self, pretrained):
        super(Encoder, self).__init__()

        # download pretrained resnet
        if pretrained and not os.path.isfile(resnet50["path"]):

            print("Downloading pretrained resnet (source : https://github.com/donnyyou/torchcv)")

            # run a shell script 
            os.system('sh models/backbones/get_pretrained_model.sh')

        model = ResNetBackbone(backbone='deepbase_resnet50_dilated8', pretrained= pretrained)

        
