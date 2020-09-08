
import math 

import torch
from torch import nn
from torch.distributions.uniform import Uniform

class MainDecoder(nn.Module):

    def __init__(self, upscale, conv_in_ch, num_classes):
        super(MainDecoder,self).__init__()
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x):
        x = self.upsample(x)
        return x


def upsample(in_channels, out_channels, upscale, kernel_size=3): 

    layers = []

    conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    nn.init.kaiming_normal_(conv1x1.weight.data, nonlinearity='relu')
    layers.append(conv1x1)

    for i in range(int(math.log(upscale, 2))):
        layers.append(PixelShuffle(out_channels, scale=2))
    return nn.Sequential(*layers)


class PixelShuffle(nn.Module):
    """
    Real-Time Single Image and Video Super-Resolution
    https://arxiv.org/abs/1609.05158
    """

    def __init__(self, n_channels, scale):

        super(PixelShuffle, self).__init__()

        self.conv = nn.Conv2d(n_channels, n_channels *(scale**2), kernel_size=1)
        icnr(self.conv.weight)

        self.shuf = nn.PixelShuffle(scale)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.shuf(self.relu(self.conv(x)))
        return x

def icnr(x, scale=2, init=nn.init.kaiming_normal_): 
    """
    Checkerboard artifact free sub-pixel convolution
    https://arxiv.org/abs/1707.02937
    """

    ni, nf, h, w = x.shape
    ni2 = int(ni/(scale**2))
    k = init(torch.zeros([ni2, nf, h, w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    k = k.contiguous().view([nf,ni,h,w]).transpose(0, 1)
    x.data.copy_(k)

class VATDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes, xi=1e-1, eps=10.0, iterations=1):
        super(VATDecoder, self).__init__()
        self.xi = xi
        self.eps = eps
        self.it = iterations
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x, _):
        r_adv = get_r_adv(x, self.upsample, self.it, self.xi, self.eps)
        x = self.upsample(x + r_adv)
        return x

class DropOutDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes, drop_rate=0.3, spatial_dropout=True):
        super(DropOutDecoder, self).__init__()
        self.dropout = nn.Dropout2d(p=drop_rate) if spatial_dropout else nn.Dropout(drop_rate)
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x, _):
        x = self.upsample(self.dropout(x))
        return x

class CutOutDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes, drop_rate=0.3, spatial_dropout=True, erase=0.4):
        super(CutOutDecoder, self).__init__()
        self.erase = erase
        self.upscale = upscale 
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x, pred=None):
        maskcut = guided_cutout(pred, upscale=self.upscale, erase=self.erase, resize=(x.size(2), x.size(3)))
        x = x * maskcut
        x = self.upsample(x)
        return x

class ContextMaskingDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes):
        super(ContextMaskingDecoder, self).__init__()
        self.upscale = upscale
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x, pred=None):
        x_masked_context = guided_masking(x, pred, resize=(x.size(2), x.size(3)),
                                          upscale=self.upscale, return_msk_context=True)
        x_masked_context = self.upsample(x_masked_context)
        return x_masked_context

class ObjectMaskingDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes):
        super(ObjectMaskingDecoder, self).__init__()
        self.upscale = upscale
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x, pred=None):
        x_masked_obj = guided_masking(x, pred, resize=(x.size(2), x.size(3)),
                                      upscale=self.upscale, return_msk_context=False)
        x_masked_obj = self.upsample(x_masked_obj)

        return x_masked_obj

class FeatureDropDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes):
        super(FeatureDropDecoder, self).__init__()
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def feature_dropout(self, x):
        attention = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
        threshold = max_val * np.random.uniform(0.7, 0.9)
        threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
        drop_mask = (attention < threshold).float()
        return x.mul(drop_mask)

    def forward(self, x, _):
        x = self.feature_dropout(x)
        x = self.upsample(x)
        return x

class FeatureNoiseDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes, uniform_range=0.3):
        super(FeatureNoiseDecoder, self).__init__()
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x, _):
        x = self.feature_based_noise(x)
        x = self.upsample(x)
        return x

