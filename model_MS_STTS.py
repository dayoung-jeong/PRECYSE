import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils as utils

class SensorTransformer(nn.Module):
    def __init__(self, datapoints = 375, features = 32, embed_dim = 768,
                 eye_features = 23, head_features = 6, ph_features = 3,
                 num_heads = 12, mlp_ratio = 4, qkv_bias = True, qk_scale = None,
                 op_type = 'cls', depth = 12, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=None, num_classes = 21, 
                 act_layer=None, init_values=None, patch_size = 1, in_chans = 4):
        
        super().__init__()
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.op_type = op_type
        
        self.token = nn.Parameter(torch.zeros(1, 1,embed_dim))

        self.eye_pos_embedding = nn.Parameter(torch.zeros(1, eye_features+1, embed_dim))
        self.head_pos_embedding = nn.Parameter(torch.zeros(1, head_features+1, embed_dim))
        self.ph_pos_embedding = nn.Parameter(torch.zeros(1, ph_features+1, embed_dim))
        
        self.pos_drop = nn.Dropout(p = drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks = nn.Sequential(*[
            utils.Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale = qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)
        
        #Classifier head
        self.fc_norm = norm_layer(embed_dim)
        self.class_head = nn.Linear(embed_dim, num_classes)
        
    def eye_forward(self, x):
        attn_total = []

        batch_size, features, embed_size = x.shape

        cls_token = torch.tile(self.token, (batch_size, 1, 1))
        
        x = torch.cat((x,cls_token),dim=1)
        x = x + self.eye_pos_embedding
        x = self.pos_drop(x)
        
        for i, blk in enumerate(self.blocks):
            x, attn = blk(x)
            attn_total.append(attn.tolist())
        
        return x
    
    def head_forward(self, x):
        attn_total = []

        batch_size, features, embed_size = x.shape

        cls_token = torch.tile(self.token, (batch_size, 1, 1))
        
        x = torch.cat((x,cls_token),dim=1)

        x = x + self.head_pos_embedding
        x = self.pos_drop(x)
        
        for i, blk in enumerate(self.blocks):
            x, attn = blk(x)
            attn_total.append(attn.tolist())
        
        return x
    
    def ph_forward(self, x):
        attn_total = []

        batch_size, features, embed_size = x.shape

        cls_token = torch.tile(self.token, (batch_size, 1, 1))
        
        x = torch.cat((x,cls_token),dim=1)

        x = x + self.ph_pos_embedding
        x = self.pos_drop(x)
        
        for i, blk in enumerate(self.blocks):
            x, attn = blk(x)
            attn_total.append(attn.tolist())
        
        return x

    def forward(self, input_x):

        batch_size, features, image_size1, image_size2, channel = input_x.shape
        x = rearrange(input_x, 'b f w h c  -> b f (w h c)', )
        
        eye = x[:,:23,:]
        head = x[:,23:29,:]
        ph = x[:,29:32,:]
        
        e_cls = self.eye_forward(eye)
        h_cls = self.head_forward(head)
        p_cls = self.ph_forward(ph)
        
        x = torch.cat((e_cls,h_cls,p_cls),dim=1)
        
        x = x[:,-1,:]
        
        x = self.fc_norm(x)

        x = self.class_head(x)
        
        return x
        