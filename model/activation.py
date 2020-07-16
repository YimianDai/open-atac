from __future__ import division
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn


class ChaATAC(HybridBlock):
    def __init__(self, channels, r, useReLU, useGlobal):
        super(ChaATAC, self).__init__()
        self.channels = channels
        self.inter_channels = int(channels // r)
        with self.name_scope():
            self.module = nn.HybridSequential(prefix='module')
            if useGlobal:
                self.module.add(nn.GlobalAvgPool2D())
            self.module.add(nn.Conv2D(self.inter_channels, kernel_size=1, strides=1, padding=0,
                                      use_bias=False))
            self.module.add(nn.BatchNorm())
            if useReLU:
                self.module.add(nn.Activation('relu'))
            self.module.add(nn.Conv2D(self.channels, kernel_size=1, strides=1, padding=0,
                                      use_bias=False))
            self.module.add(nn.BatchNorm())
            self.module.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, x):

        wei = self.module(x)
        x = F.broadcast_mul(x, wei)

        return x
