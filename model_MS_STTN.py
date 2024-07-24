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
    def __init__(self, datapoints = 375, features = 32, eye_features = 23, head_features = 6, ph_features = 3,
                 embed_dim = 192, num_heads = 8, mlp_ratio = 4, qkv_bias = True, qk_scale = None,
                 op_type = 'cls', depth = 12, sdepth = 6, tdepth = 6, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=None, num_classes = 21, 
                 act_layer=None, init_values=None, patch_size = 1, in_chans = 1, spatial_embed=32):
        
        super().__init__()
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.op_type = op_type
        self.token = nn.Parameter(torch.zeros(1,1,embed_dim))

        temporal_embed = spatial_embed * features
        eye_temporal_embed = spatial_embed * eye_features
        head_temporal_embed = spatial_embed * head_features
        ph_temporal_embed = spatial_embed * ph_features
        
        self.eye_temporal_pos_embedding = nn.Parameter(torch.zeros(1, datapoints+1, eye_temporal_embed))
        self.head_temporal_pos_embedding = nn.Parameter(torch.zeros(1, datapoints+1, head_temporal_embed))
        self.ph_temporal_pos_embedding = nn.Parameter(torch.zeros(1, datapoints+1, ph_temporal_embed))
        
        self.pos_drop = nn.Dropout(p = drop_rate)
        
        self.spatial_patch_embedding = nn.Linear(in_chans, spatial_embed)
        
        self.eye_spatial_pos_embedding = nn.Parameter(torch.zeros(1, eye_features+1, spatial_embed))
        self.head_spatial_pos_embedding = nn.Parameter(torch.zeros(1, head_features+1, spatial_embed))
        self.ph_spatial_pos_embedding = nn.Parameter(torch.zeros(1, ph_features+1, spatial_embed))
        
        self.spatial_token = nn.Parameter(torch.zeros(1,1,spatial_embed))

        self.eye_spatial_to_temporal_token = nn.Linear(datapoints*spatial_embed, eye_features*spatial_embed)
        self.head_spatial_to_temporal_token = nn.Linear(datapoints*spatial_embed, head_features*spatial_embed)
        self.ph_spatial_to_temporal_token = nn.Linear(datapoints*spatial_embed, ph_features*spatial_embed)
        
        tdpr = [x.item() for x in torch.linspace(0, drop_path_rate, tdepth)]
        sdpr = [x.item() for x in torch.linspace(0, drop_path_rate, sdepth)]
        
        self.eye_temporal_blocks = nn.Sequential(*[
            utils.Block(
                dim=eye_temporal_embed, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale = qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path=tdpr[i], norm_layer=norm_layer)
            for i in range(tdepth)])
        self.head_temporal_blocks = nn.Sequential(*[
            utils.Block(
                dim=head_temporal_embed, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale = qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path=tdpr[i], norm_layer=norm_layer)
            for i in range(tdepth)])
        self.ph_temporal_blocks = nn.Sequential(*[
            utils.Block(
                dim=ph_temporal_embed, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale = qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path=tdpr[i], norm_layer=norm_layer)
            for i in range(tdepth)])
        
        self.spatial_blocks = nn.Sequential(*[
            utils.Block(
                dim=spatial_embed, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale = qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path=sdpr[i], norm_layer=norm_layer)
            for i in range(sdepth)])

        self.eye_temporal_norm = norm_layer(eye_temporal_embed)
        self.head_temporal_norm = norm_layer(head_temporal_embed)
        self.ph_temporal_norm = norm_layer(ph_temporal_embed)
        self.spatial_norm = norm_layer(spatial_embed)
        
        #Classifier head
        self.fc_norm = norm_layer(temporal_embed)
        self.class_head = nn.Linear(temporal_embed, num_classes)
    
    def spatial_forward(self, x):
        
        batch_size, datapoints, features, channel = x.shape
        
        x = rearrange(x, 'b f p c  -> (b f) p c', )

        x = self.spatial_patch_embedding(x)
        
        cls_token = torch.tile(self.spatial_token, (batch_size * datapoints, 1, 1))
        

        x = torch.cat((x,cls_token),dim=1)
        
        if features == 23:
            x = x + self.eye_spatial_pos_embedding
        elif features == 6:
            x = x + self.head_spatial_pos_embedding
        elif features == 3:
            x = x + self.ph_spatial_pos_embedding
        
        x = self.pos_drop(x)

        x = self.spatial_blocks(x)
        
        x = self.spatial_norm(x)

        spatial_embed_dim = x.shape[-1]

        cls_token = x[:,-1,:]

        cls_token = torch.reshape(cls_token, (batch_size,datapoints*spatial_embed_dim))

        x = x[:,:features,:]
        x = rearrange(x, '(b f) p Se-> b f (p Se)', f=datapoints)
        
        return x, cls_token
    
    def eye_forward(self, x):
        
        x, cls_token = self.spatial_forward(x)
        
        cls_token = self.eye_spatial_to_temporal_token(cls_token)
        
        cls_token = torch.unsqueeze(cls_token,dim=1)

        batch_size, datapoints, channel = x.shape
        
        x = torch.cat((x,cls_token), dim = 1)
        
        x = x + self.eye_temporal_pos_embedding
        x = self.pos_drop(x)

        x = self.eye_temporal_blocks(x)
        
        x = self.eye_temporal_norm(x)

        cls_token = x[:,-1,:]

        cls_token = cls_token.view(batch_size,-1)
        
        return cls_token
    
    def head_forward(self, x):
        
        x, cls_token = self.spatial_forward(x)
        
        cls_token = self.head_spatial_to_temporal_token(cls_token)
        
        cls_token = torch.unsqueeze(cls_token,dim=1)

        batch_size, datapoints, channel = x.shape
        
        x = torch.cat((x,cls_token), dim = 1)
        
        x = x + self.head_temporal_pos_embedding
        x = self.pos_drop(x)

        x = self.head_temporal_blocks(x)
        
        x = self.head_temporal_norm(x)

        cls_token = x[:,-1,:]

        cls_token = cls_token.view(batch_size,-1)
        
        return cls_token
    
    def ph_forward(self, x):
        
        x, cls_token = self.spatial_forward(x)
        
        cls_token = self.ph_spatial_to_temporal_token(cls_token)
        
        cls_token = torch.unsqueeze(cls_token,dim=1)

        batch_size, datapoints, channel = x.shape
        
        x = torch.cat((x,cls_token), dim = 1)
        
        x = x + self.ph_temporal_pos_embedding
        x = self.pos_drop(x)

        x = self.ph_temporal_blocks(x)
        
        x = self.ph_temporal_norm(x)

        cls_token = x[:,-1,:]

        cls_token = cls_token.view(batch_size,-1)
        
        return cls_token
        
    def forward(self, input_x):
        
        eye = input_x[:,:,0:23]
        head = input_x[:,:,23:29]
        ph = input_x[:,:,29:32]
        
        e_cls = self.eye_forward(eye.unsqueeze(3))
        h_cls = self.head_forward(head.unsqueeze(3))
        p_cls = self.ph_forward(ph.unsqueeze(3))
        
        x = torch.cat((e_cls,h_cls,p_cls), dim=1)
        
        x = self.fc_norm(x)

        x = self.class_head(x)
        
        return x
        