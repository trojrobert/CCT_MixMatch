import numpy as np 
import logging
import torch.nn as nn 


class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self):
        raise NotImplementedError