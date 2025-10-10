#!/usr/bin/env python3

import torch
from torch import nn

import os, json


def getConfig(config_name, base_config):
    config = {}
    if os.path.exists(config_name):
        with open(config_name, 'r') as fp:
            config = json.load(fp)
    else:
        config.update(base_config)
        json.dump( config, open(config_name, 'w'), indent = 1 )
    return config

class SkipLayer(nn.Module):
    """
        Three conv3d layers, two consecutive and one that skips.
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.a = nn.Sequential(
            nn.Conv3d( in_c, out_c//2, 3, padding="same" ),
            nn.Conv3d( out_c//2, out_c//2, 3, padding="same"),
            nn.ReLU() )
        self.b = nn.Sequential( nn.Conv3d( in_c, out_c//2, 3, padding="same" ), nn.ReLU() )

    def forward(self, x):
        n0 = self.a(x)
        n1 = self.b(x)
        return torch.cat([n0, n1], axis=1)
