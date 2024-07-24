# Copyright (c) CryoNet Team, and its affiliates. All Rights Reserved

"""
Backbone modules.
"""
import math
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch import nn
from cryofold.utils import checkpoint_utils as ckp
from cryofold.model.resnet import *
from cryofold.model.primitives import LayerNorm3d, NestedTensor

class PatchEmbedding(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, replace_stride_with_dilation, pretrained=False, args=None):
        super().__init__()
        self.args = args
        self.b1_c = args.patch_c1
        self.b2_c = args.patch_c2
        self.inplanes = args.patch_c2
        self.relu0 = nn.ReLU(inplace=True)

        self.block1 = nn.Sequential(
            nn.Conv3d(1, self.b1_c, kernel_size=2, stride=2),
            LayerNorm3d(self.b1_c),
            nn.ReLU(inplace=True)
        )

        self.block2 = nn.Sequential(
            nn.Conv3d(self.b1_c, self.b2_c, kernel_size=2, stride=2),
            LayerNorm3d(self.b2_c),
            nn.ReLU(inplace=True)
        )        

    def forward(self, x, **kwargs):
        # B, C, D, H, W = x.shape
        x = self.relu0(x)
        if self.training and x.requires_grad:
            x2 = ckp.checkpoint(self.block1, x)
        else:
            x2 = self.block1(x) #(1, 256, W/2, W/2, W/2)
        if self.training and x2.requires_grad:
            x = ckp.checkpoint(self.block2, x2)
        else:
            x = self.block2(x2) #(11, 256, W/4, W/4, W/4)
        
        return x, x2


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list):
        x = tensor_list.tensors
        s=x.shape
        # mask = tensor_list.mask
        not_mask = torch.ones((1, s[2], s[3], s[4]), dtype=torch.bool, device=x.device)
        # not_mask = ~mask
        # import pdb;pdb.set_trace()
        x_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        z_embed = not_mask.cumsum(3, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            x_embed = x_embed / (x_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
            z_embed = z_embed / (z_embed[:, :, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t / 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, :, None] / dim_t
        pos_z = z_embed[:, :, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_z = torch.stack((pos_z[:, :, :, :, 0::2].sin(), pos_z[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos = torch.cat((pos_x, pos_y, pos_z), dim=4).permute(0, 4, 1, 2, 3)
        return pos

class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, 
                 train_backbone: bool, 
                 num_channels: int, 
                 return_interm_layers: bool,
                 args):
        super().__init__()
        # for name, parameter in backbone.named_parameters():
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        self.args = args
        # if return_interm_layers:
        #     return_layers = {"layer1": "0","layer2": "1", "layer3": "2", "layer4": "3"}
        # else:
        #     return_layers = {'layer4': "0"}
        
        # self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.body = backbone
        self.num_channels = num_channels
        self.num_channels1 = num_channels//4
        self.num_channels2 = num_channels

    def forward(self, tensor_list):
        if self.args.frozen_backbone:
            with torch.no_grad():
                xs, xs2 = self.body(tensor_list.tensors) 
        else:
            xs, xs2 = self.body(tensor_list.tensors)
        if tensor_list.mask is not None:
            # mask  = F.interpolate(tensor_list.mask[None,None].float(),  size=xs.shape[-3:]).type(torch.bool)[0]
            # mask2 = F.interpolate(tensor_list.mask[None,None].float(), size=xs2.shape[-3:]).type(torch.bool)[0]
            # import pdb;pdb.set_trace()
            mask  = F.interpolate(tensor_list.mask.float(),  size=xs.shape[-3:]).type(torch.bool).float()
            mask2 = F.interpolate(tensor_list.mask.float(), size=xs2.shape[-3:]).type(torch.bool).float()
        else:
            mask = None
        out: Dict[str, NestedTensor] = {}
        if self.args.density_head:
            out['0'] = NestedTensor(xs2, mask2)
        out['1'] = NestedTensor(xs, mask)
        
        return out

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 args
                 ):
        backbone = globals()[name](
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=False, args=args)
        num_channels = backbone.inplanes # 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers, args=args)
        if args.frozen_backbone:
            for p in self.parameters():
                p.requires_grad_(False)

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x))
            # pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos

def build_backbone(args):
    N_steps = args.hidden_dim // 3
    position_embedding = PositionEmbeddingSine(N_steps, normalize=True)

    train_backbone = True
    return_interm_layers = False

    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation, args)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    model.num_channels1 = backbone.num_channels1
    model.num_channels2 = backbone.num_channels2
    return model
