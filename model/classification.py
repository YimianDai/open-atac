from __future__ import division
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from mxnet.gluon.block import HybridBlock
from .activation import ChaATAC


def _conv3x3(channels, stride, in_channels):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=in_channels)


class ResBlockV2ATAC(HybridBlock):
    def __init__(self, act_type, r, skernel, dilation, channels, useReLU, useGlobal, asBackbone,
                 stride, downsample=False, in_channels=0, norm_layer=BatchNorm,
                 norm_kwargs=None, **kwargs):
        super(ResBlockV2ATAC, self).__init__(**kwargs)
        self.bn1 = norm_layer(**({} if norm_kwargs is None else norm_kwargs))
        self.conv1 = _conv3x3(channels, stride, in_channels)
        self.bn2 = norm_layer(**({} if norm_kwargs is None else norm_kwargs))
        self.conv2 = _conv3x3(channels, 1, channels)
        if downsample:
            self.downsample = nn.Conv2D(channels, 1, stride, use_bias=False, in_channels=in_channels)
        else:
            self.downsample = None

        if act_type == 'relu':
            self.msAA1 = nn.Activation('relu')
            self.msAA2 = nn.Activation('relu')
        elif act_type == 'prelu':
            self.msAA1 = nn.PReLU()
            self.msAA2 = nn.PReLU()
        elif act_type == 'elu':
            self.msAA1 = nn.ELU()
            self.msAA2 = nn.ELU()
        elif act_type == 'selu':
            self.msAA1 = nn.SELU()
            self.msAA2 = nn.SELU()
        elif act_type == 'gelu':
            self.msAA1 = nn.GELU()
            self.msAA2 = nn.GELU()
        elif act_type == 'swish':
            self.msAA1 = nn.Swish()
            self.msAA2 = nn.Swish()
        elif act_type == 'ChaATAC':
            self.msAA1 = ChaATAC(channels=in_channels, r=r, useReLU=useReLU, useGlobal=useGlobal)
            self.msAA2 = ChaATAC(channels=channels, r=r, useReLU=useReLU, useGlobal=useGlobal)
        else:
            raise ValueError("Unknown act_type in ResBlockV2ATAC")

    def hybrid_forward(self, F, x):
        """Hybrid forward"""
        residual = x

        x = self.bn1(x)
        x = self.msAA1(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.msAA2(x)
        x = self.conv2(x)

        if self.downsample:
            residual = self.downsample(residual)
        return x + residual


class ResNet20V2ATAC(HybridBlock):
    def __init__(self, layers, channels, classes,
                 act_type, r, skernel, dilation, useReLU, useGlobal, act_layers, replace_act,
                 act_order, asBackbone, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(ResNet20V2ATAC, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(norm_layer(scale=False, center=False,
                                         **({} if norm_kwargs is None else norm_kwargs)))
            self.features.add(nn.Conv2D(channels[0], 3, 1, 1, use_bias=False))

            in_channels = channels[0]
            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                if act_order == 'bac':
                    if i + act_layers < len(channels):
                        tmp_act_type = replace_act
                    else:
                        tmp_act_type = act_type
                elif act_order == 'pre':
                    if i + 1 > act_layers:
                        tmp_act_type = replace_act
                    else:
                        tmp_act_type = act_type
                else:
                    raise ValueError('Unknown act_order')
                self.features.add(self._make_layer(
                    layers=num_layer, channels=channels[i+1], in_channels=in_channels,
                    stride=stride, stage_index=i+1, act_type=tmp_act_type, r=r, skernel=skernel,
                    dilation=dilation, useReLU=useReLU, useGlobal=useGlobal,
                    asBackbone=asBackbone, norm_layer=norm_layer, norm_kwargs=norm_kwargs
                ))
                in_channels = channels[i+1]

            self.features.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))

            if act_order == 'bac':
                if act_layers <= 0:
                    tmp_act_type = replace_act
                else:
                    tmp_act_type = act_type
            elif act_order == 'pre':
                if act_layers >= 4:
                    tmp_act_type = act_type
                else:
                    tmp_act_type = replace_act
            else:
                raise ValueError('Unknown act_order')

            if tmp_act_type == 'relu':
                self.features.add(nn.Activation('relu'))
            elif tmp_act_type == 'prelu':
                self.features.add(nn.PReLU())
            elif tmp_act_type == 'elu':
                self.features.add(nn.ELU())
            elif tmp_act_type == 'selu':
                self.features.add(nn.SELU())
            elif tmp_act_type == 'gelu':
                self.features.add(nn.GELU())
            elif tmp_act_type == 'swish':
                self.features.add(nn.Swish())
            elif tmp_act_type == 'ChaATAC':
                self.features.add(ChaATAC(channels=in_channels, r=r, useReLU=useReLU,
                                          useGlobal=useGlobal))
            else:
                raise ValueError("Unknown act_type in ResBlockV2ATAC")

            self.features.add(nn.GlobalAvgPool2D())
            self.features.add(nn.Flatten())
            self.output = nn.Dense(classes, in_units=in_channels)

    def _make_layer(self, layers, channels, in_channels, stride, stage_index,
                    act_type, r, skernel, dilation, useReLU, useGlobal, asBackbone,
                    norm_layer=BatchNorm, norm_kwargs=None):
        layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():
            layer.add(ResBlockV2ATAC(act_type, r, skernel=skernel, dilation=dilation,
                                     channels=channels, useReLU=useReLU, useGlobal=useGlobal,
                                     asBackbone=asBackbone, stride=stride,
                                     downsample=channels != in_channels,
                                     in_channels=in_channels, prefix='',
                                     norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            for bidx in range(layers-1):
                layer.add(ResBlockV2ATAC(act_type, r, skernel=skernel, dilation=dilation,
                                         channels=channels, useReLU=useReLU,
                                         useGlobal=useGlobal, asBackbone=asBackbone,
                                         stride=1, downsample=False,
                                         in_channels=channels, prefix='',
                                         norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


class ResNet50_v1bATAC(HybridBlock):
    def __init__(self, act_type, r, act_layers, layers, classes=1000, dilated=False, norm_layer=BatchNorm,
                 norm_kwargs=None, last_gamma=False, deep_stem=False, stem_width=32,
                 avg_down=False, final_drop=0.0, use_global_stats=False,
                 name_prefix='', **kwargs):
        self.inplanes = stem_width*2 if deep_stem else 64
        super(ResNet50_v1bATAC, self).__init__(prefix=name_prefix)
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        if use_global_stats:
            norm_kwargs['use_global_stats'] = True
        self.norm_kwargs = norm_kwargs

        with self.name_scope():
            self.conv1 = nn.Conv2D(channels=64, kernel_size=7, strides=2,
                                   padding=3, use_bias=False)
            self.bn1 = norm_layer(in_channels=64 if not deep_stem else stem_width*2,
                                  **norm_kwargs)
            self.relu = nn.Activation('relu')
            self.maxpool = nn.MaxPool2D(pool_size=3, strides=2, padding=1)

            if act_layers >= len(layers):
                tmp_act_type = act_type
            else:
                tmp_act_type = 'relu'
            self.layer1 = self._make_layer(tmp_act_type, r, 1, BottleneckV1bATAC, 64, layers[0],
                                           avg_down=avg_down, norm_layer=norm_layer,
                                           last_gamma=last_gamma)
            if act_layers >= len(layers) - 1:
                tmp_act_type = act_type
            else:
                tmp_act_type = 'relu'
            self.layer2 = self._make_layer(tmp_act_type, r, 2, BottleneckV1bATAC, 128, layers[1],
                                           strides=2, avg_down=avg_down,
                                           norm_layer=norm_layer, last_gamma=last_gamma)
            if dilated:
                if act_layers >= len(layers) - 2:
                    tmp_act_type = act_type
                else:
                    tmp_act_type = 'relu'
                self.layer3 = self._make_layer(tmp_act_type, r, 3, BottleneckV1bATAC, 256, layers[2],
                                               strides=1, dilation=2, avg_down=avg_down,
                                               norm_layer=norm_layer, last_gamma=last_gamma)
                if act_layers >= len(layers) - 3:
                    tmp_act_type = act_type
                else:
                    tmp_act_type = 'relu'
                self.layer4 = self._make_layer(tmp_act_type, r, 4, BottleneckV1bATAC, 512, layers[3],
                                               strides=1, dilation=4, avg_down=avg_down,
                                               norm_layer=norm_layer, last_gamma=last_gamma)
            else:
                if act_layers >= len(layers) - 2:
                    tmp_act_type = act_type
                else:
                    tmp_act_type = 'relu'
                self.layer3 = self._make_layer(tmp_act_type, r, 3, BottleneckV1bATAC, 256, layers[2],
                                               strides=2, avg_down=avg_down,
                                               norm_layer=norm_layer, last_gamma=last_gamma)
                if act_layers >= len(layers) - 3:
                    tmp_act_type = act_type
                else:
                    tmp_act_type = 'relu'
                self.layer4 = self._make_layer(tmp_act_type, r, 4, BottleneckV1bATAC, 512, layers[3],
                                               strides=2, avg_down=avg_down,
                                               norm_layer=norm_layer, last_gamma=last_gamma)

            self.avgpool = nn.GlobalAvgPool2D()
            self.flat = nn.Flatten()
            self.drop = None
            if final_drop > 0.0:
                self.drop = nn.Dropout(final_drop)
            self.fc = nn.Dense(in_units=512 * BottleneckV1bATAC.expansion, units=classes)

    def _make_layer(self, act_type, r, stage_index, block, planes, blocks, strides=1, dilation=1,
                    avg_down=False, norm_layer=None, last_gamma=False):
        downsample = None
        if strides != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.HybridSequential(prefix='down%d_'%stage_index)
            with downsample.name_scope():
                if avg_down:
                    if dilation == 1:
                        downsample.add(nn.AvgPool2D(pool_size=strides, strides=strides,
                                                    ceil_mode=True, count_include_pad=False))
                    else:
                        downsample.add(nn.AvgPool2D(pool_size=1, strides=1,
                                                    ceil_mode=True, count_include_pad=False))
                    downsample.add(nn.Conv2D(channels=planes * block.expansion, kernel_size=1,
                                             strides=1, use_bias=False))
                    downsample.add(norm_layer(in_channels=planes * block.expansion,
                                              **self.norm_kwargs))
                else:
                    downsample.add(nn.Conv2D(channels=planes * block.expansion,
                                             kernel_size=1, strides=strides, use_bias=False))
                    downsample.add(norm_layer(in_channels=planes * block.expansion,
                                              **self.norm_kwargs))

        layers = nn.HybridSequential(prefix='layers%d_'%stage_index)
        with layers.name_scope():
            if dilation in (1, 2):
                layers.add(BottleneckV1bATAC(
                    act_type, r, planes, strides, dilation=1, downsample=downsample,
                    previous_dilation=dilation, norm_layer=norm_layer,
                    norm_kwargs=self.norm_kwargs, last_gamma=last_gamma))
            elif dilation == 4:
                layers.add(BottleneckV1bATAC(
                    act_type, r, planes, strides, dilation=2, downsample=downsample,
                    previous_dilation=dilation, norm_layer=norm_layer,
                    norm_kwargs=self.norm_kwargs, last_gamma=last_gamma))
            else:
                raise RuntimeError("=> unknown dilation size: {}".format(dilation))

            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.add(BottleneckV1bATAC(
                    act_type, r, planes, dilation=dilation, previous_dilation=dilation,
                    norm_layer=norm_layer, norm_kwargs=self.norm_kwargs,
                    last_gamma=last_gamma))

        return layers

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flat(x)
        if self.drop is not None:
            x = self.drop(x)
        x = self.fc(x)

        return x


class BottleneckV1bATAC(HybridBlock):
    """ResNetV1b BottleneckV1b
    """
    # pylint: disable=unused-argument
    expansion = 4
    def __init__(self, act_type, r, planes, strides=1, dilation=1,
                 downsample=None, previous_dilation=1, norm_layer=None,
                 norm_kwargs=None, last_gamma=False, **kwargs):
        super(BottleneckV1bATAC, self).__init__()
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        self.conv1 = nn.Conv2D(channels=planes, kernel_size=1,
                               use_bias=False)
        self.bn1 = norm_layer(in_channels=planes, **norm_kwargs)
        self.conv2 = nn.Conv2D(channels=planes, kernel_size=3, strides=strides,
                               padding=dilation, dilation=dilation, use_bias=False)
        self.bn2 = norm_layer(in_channels=planes, **norm_kwargs)
        self.conv3 = nn.Conv2D(channels=planes * 4, kernel_size=1, use_bias=False)
        if not last_gamma:
            self.bn3 = norm_layer(in_channels=planes*4, **norm_kwargs)
        else:
            self.bn3 = norm_layer(in_channels=planes*4, gamma_initializer='zeros',
                                  **norm_kwargs)

        self.downsample = downsample
        self.dilation = dilation
        self.strides = strides

        if act_type == 'relu':
            self.relu1 = nn.Activation('relu')
            self.relu2 = nn.Activation('relu')
            self.relu3 = nn.Activation('relu')
        elif act_type == 'ChaATAC':
            self.relu1 = ChaATAC(channels=planes,  r=r, useReLU=True, useGlobal=False)
            self.relu2 = ChaATAC(channels=planes,  r=r, useReLU=True, useGlobal=False)
            self.relu3 = nn.Activation('relu')

    def hybrid_forward(self, F, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu3(out)

        return out
