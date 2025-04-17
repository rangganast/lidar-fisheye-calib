import torch.nn as nn
from spherenet import SphereConv2D

def sphereconv(in_ch, out_ch, strd):
    conv = SphereConv2D(in_ch, out_ch, stride=strd)
    nn.init.kaiming_uniform_(conv.weight, mode='fan_out', nonlinearity='relu')
    return conv

def sphereconv1x1(in_ch, out_ch, strd): #spherconv biasa ini mah gabisa dibikin 1x1 ngab
    conv = SphereConv2D(in_ch, out_ch, stride=strd)
    nn.init.kaiming_uniform_(conv.weight, mode='fan_out', nonlinearity='relu')
    return conv

def atrconv1x1(in_ch, out_ch, strd, pad, dil):
    conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=strd, padding=pad, dilation=dil)
    nn.init.kaiming_uniform_(conv.weight, mode='fan_out', nonlinearity='relu')
    return conv


class SphereConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super(SphereConvBNReLU, self).__init__()
        self.conv = sphereconv1x1(in_ch, out_ch, stride)
        self.BN = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.05)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.BN(x)
        x = self.dropout(x)
        return x
    
class DilatedSphereConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super(DilatedSphereConvBNReLU, self).__init__()
        self.conv = dilatedsphereconv(in_ch, out_ch, stride, dil=1)
        self.BN = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.05)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.BN(x)
        x = self.dropout(x)
        return x