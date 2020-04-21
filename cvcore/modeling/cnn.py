import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
import os

import torchvision
import timm
from timm.models.layers.activations import Swish, Mish
from timm.models.layers.adaptive_avgmax_pool import SelectAdaptivePool2d


class NormSoftmax(nn.Module):
    def __init__(self, in_features, out_features, temperature=1.):
        super(NormSoftmax, self).__init__()
        self.weight = nn.Parameter(
            torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight.data)

        self.ln = nn.LayerNorm(in_features, elementwise_affine=False)
        self.temperature = nn.Parameter(torch.Tensor([temperature]))

    def forward(self, x):
        x = self.ln(x)
        x = torch.matmul(F.normalize(x), F.normalize(self.weight))
        x = x / self.temperature
        return x


def convert_swish_to_mish(model):
    for child_name, child in model.named_children():
        if isinstance(child, Swish):
            setattr(model, child_name, Mish(inplace=True))
        else:
            convert_swish_to_mish(child)


def convert_relu_to_mish(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, Mish(inplace=True))
        else:
            convert_relu_to_mish(child)


class EfficientNet(nn.Module):
    """
    EfficientNet B0-B8.
    Args:
        cfg (CfgNode): configs
    """
    def __init__(self, cfg):
        super(EfficientNet, self).__init__()
        model_name = cfg.MODEL.NAME
        pretrained = cfg.MODEL.PRETRAINED
        input_channels = cfg.DATA.INP_CHANNEL
        pool_type = cfg.MODEL.POOL_TYPE
        drop_connect_rate = cfg.MODEL.DROP_CONNECT
        self.drop_rate = cfg.MODEL.DROPOUT
        num_w_classes = cfg.MODEL.NUM_CLASSES

        backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=input_channels,
            drop_connect_rate=drop_connect_rate,
        )
        self.conv_stem = backbone.conv_stem
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        ### Original blocks ###
        for i in range(len((backbone.blocks))):
            setattr(self, "block{}".format(str(i)), backbone.blocks[i])
        self.conv_head = backbone.conv_head
        self.bn2 = backbone.bn2
        self.act2 = backbone.act2
        self.global_pool = SelectAdaptivePool2d(pool_type=pool_type)
        self.num_features = backbone.num_features * self.global_pool.feat_mult()
        ### Baseline head ###
        if cfg.MODEL.CLS_HEAD == "linear":
            self.w_fc = nn.Linear(self.num_features, num_w_classes)
        elif cfg.MODEL.CLS_HEAD == "norm":
            self.w_fc = NormSoftmax(self.num_features, num_w_classes)
        # Replace with Mish activation
        if cfg.MODEL.ACTIVATION == "mish":
            convert_swish_to_mish(self)
        del backbone

    def _features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x

    def forward(self, x):
        x = self._features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        w_logits = self.w_fc(x)
        return w_logits

class ResNet(nn.Module):
    """
    Generic ResNets.
    Args:
        cfg (CfgNode): configs
    """
    def __init__(self, cfg):
        super(ResNet, self).__init__()
        model_name = cfg.MODEL.NAME
        pretrained = cfg.MODEL.PRETRAINED
        input_channels = cfg.DATA.INP_CHANNEL
        pool_type = cfg.MODEL.POOL_TYPE
        self.drop_rate = cfg.MODEL.DROPOUT
        num_w_classes = cfg.MODEL.NUM_CLASSES

        backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=input_channels
        )
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        ### Original blocks ###
        self.block1 = backbone.layer1
        self.block2 = backbone.layer2
        self.block3 = backbone.layer3
        self.block4 = backbone.layer4
        self.global_pool = SelectAdaptivePool2d(pool_type=pool_type)
        self.num_features = backbone.num_features * self.global_pool.feat_mult()
        ### Baseline head ###
        if cfg.MODEL.CLS_HEAD == "linear":
            self.w_fc = nn.Linear(self.num_features, num_w_classes)
        elif cfg.MODEL.CLS_HEAD == "norm":
            self.w_fc = NormSoftmax(self.num_features, num_w_classes)
        # Replace with Mish activation
        if cfg.MODEL.ACTIVATION == "mish":
            convert_relu_to_mish(self)
        del backbone

    def _features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x

    def forward(self, x):
        x = self._features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        w_logits = self.w_fc(x)
        return w_logits


class DenseNet(nn.Module):
    """
    Generic DenseNets.
    Args:
        cfg (CfgNode): configs
    """
    def __init__(self, cfg):
        super(DenseNet, self).__init__()
        model_name = cfg.MODEL.NAME
        pretrained = cfg.MODEL.PRETRAINED
        input_channels = cfg.DATA.INP_CHANNEL
        pool_type = cfg.MODEL.POOL_TYPE
        self.drop_rate = cfg.MODEL.DROPOUT
        num_w_classes = cfg.MODEL.NUM_CLASSES

        backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=input_channels
        )

        ### Original blocks ###
        self.features = backbone.features
        self.global_pool = SelectAdaptivePool2d(pool_type=pool_type)
        self.num_features = backbone.features.norm5.num_features * self.global_pool.feat_mult()
        ### Baseline head ###
        if cfg.MODEL.CLS_HEAD == "linear":
            self.w_fc = nn.Linear(self.num_features, num_w_classes)
        elif cfg.MODEL.CLS_HEAD == "norm":
            self.w_fc = NormSoftmax(self.num_features, num_w_classes)
        if cfg.MODEL.ACTIVATION == "mish":
            convert_relu_to_mish(self)
        del backbone

    def _features(self, x):
        x = self.features(x)
        return x

    def forward(self, x):
        x = self._features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        w_logits = self.w_fc(x)
        return w_logits