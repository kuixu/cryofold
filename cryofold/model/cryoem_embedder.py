# Copyright (c) CryoFold Team, and its affiliates. All Rights Reserved

import os
from tkinter.messagebox import NO
import torch
import torch.nn.functional as F
from torch import nn
from torch import nn, Tensor
import copy
import random
from typing import Optional, List

import pytorch_lightning as pl

from cryofold.utils import checkpoint_utils as ckp
from cryofold.model.primitives import LayerNorm, NestedTensor
from cryofold.model.gattention import GaussianMultiheadAttention
from cryofold.model.backbone import build_backbone

DTYPE = torch.get_default_dtype()
# DTYPE = torch.bfloat16


class CryoformerEmbedder(pl.LightningModule):
    """ This is the CryoNet module that performs Model building from cryo-EM density maps """
    def __init__(self, args, **kwargs):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            aa_classes: number of amino acids classes
            ss_classes: number of secondary structure classes
        """
        super().__init__()
        # ["cryoformer_embedder"]
        self.args        = args
        self.args['dtype'] = DTYPE
        self.backbone    = build_backbone(args)
        self.proj  = nn.Conv3d(self.backbone.num_channels, args.hidden_dim, kernel_size=1)
        
    def forward(self, batch: dict):
        density = batch['cryoem_density']
        mask = batch['cryoem_mask']
        # import pdb;pdb.set_trace()

        if len(density.shape)==4:
            density = density[None] 
            mask    = batch['cryoem_mask'][None]
        # else:
        #     print("wrong shape?")
        #     import pdb;pdb.set_trace()
            
        density_tensors = NestedTensor(density, mask)
        # density = NestedTensor(batch['density'][None].to(DTYPE), batch['density_mask'][0].to(DTYPE))
        # import pdb;pdb.set_trace()

        den_feats, den_poss = self.backbone(density_tensors)
        # import pdb;pdb.set_trace()

        den_feat0, den_mask0 = den_feats[0].decompose()
        den_feat1, den_mask1 = den_feats[1].decompose()
        den_pos = den_poss[-1].to(den_feat1)
        den_wei = F.max_pool3d(density,  kernel_size=4)

        den_repr = self.proj(den_feat1)
        # import pdb;pdb.set_trace()

        den_repr_dict ={
            "density_feats": [den_feat0, den_feat1],
            "density_repr": den_repr,
            "density_mask": den_mask1,
            "density_pos": den_pos,
            "density_wei": den_wei,
        }
        # .to(DTYPE)
        return den_repr_dict

class CryoformerEncoder(pl.LightningModule):
    def __init__(self, d_model=384, no_head=8, no_blocks=8, dim_feedforward=2048, 
                dropout=0.1, activation= "relu", **kwargs):
        super().__init__()
        """
        "d_model":384, 
        "no_head":8, 
        "no_blocks":8,
        "dim_feedforward":2048,
        "dropout":0.1,
        "activation": "relu",
        """
        self.no_blocks = no_blocks
        encoder_layer = CryoformerEncoderLayer(
            d_model, no_head, dim_feedforward, dropout, activation)
        self.layers = _get_clones(encoder_layer, self.no_blocks)
        # if args.frozen_encoder:
        #     for p in self.parameters():
        #         p.requires_grad_(False)

    def forward(self, density_repr,
                density_mask: Optional[Tensor] = None,
                density_pos: Optional[Tensor] = None,
                density_wei: Optional[Tensor] = None,):
        output = density_repr

        for layer in self.layers:
            if self.training and output.requires_grad:
                output = ckp.checkpoint(layer, output, density_mask, density_pos, density_wei)
            else:
                output = layer(output, density_mask, density_pos, density_wei)
        return output
        
class CryoformerDecoder(pl.LightningModule):
    def __init__(self, d_model=384, no_head=8, no_blocks=8, dim_feedforward=2048, dropout=0.1, 
                activation= "relu", return_intermediate=False, **kwargs):
        super().__init__()
        """
        "d_model":384, 
        "no_head":8, 
        "no_blocks": 8,
        "dim_feedforward":2048,
        "dropout":0.1,
        "activation": "relu",
        "return_intermediate_dec": True
        """
        # self.layers = _get_clones(decoder_layer, num_layers)
        decoder_layer = CryoformerDecoderLayer(
                d_model, no_head, dim_feedforward, dropout, activation)
        self.layers = _get_clones(decoder_layer, no_blocks)
        self.no_blocks = no_blocks
        self.norm = LayerNorm(d_model)
        self.return_intermediate = return_intermediate

    
    def forward(self, tgt_single, tgt_msa, tgt_pair,
                aa_embed, density_repr, density_mask, density_pos, density_wei
                ):
        if self.return_intermediate:
            intermediate = {
                'sgl_repr': [],
                # 'coa_repr': [],
                # 'msa_repr': [],
                # 'par_repr': [],
            }
        # out = [tgt_single, tgt_msa, tgt_pair] 
        out = tgt_single
        decoder_param = [
            tgt_single, tgt_msa, tgt_pair, 
            aa_embed, density_repr, density_mask, density_pos, density_wei]
        i = 0
        for layer in self.layers:
            if self.training and tgt_single.requires_grad:
                out = ckp.checkpoint(layer, *decoder_param)
            else:
                out = layer(*decoder_param)

            decoder_param[0] = out

            if self.return_intermediate:
                intermediate['sgl_repr'].append(self.norm(out))
                # intermediate['sgl_repr'].append(out[0])
                # if len(out)==4:
                #     intermediate['coa_repr'].append(out[3])
                # intermediate['msa_repr'].append(out[1])
                # intermediate['par_repr'].append(out[2])
            i += 1
        if self.norm is not None:
            output = self.norm(out)
            if self.return_intermediate:
                intermediate['sgl_repr'].pop()
                intermediate['sgl_repr'].append(output)
        
        if self.return_intermediate:
            # return torch.stack(intermediate['sgl_repr']), torch.stack(intermediate['msa_repr']), torch.stack(intermediate['par_repr']) 
            # return torch.stack(intermediate['sgl_repr']), out[1][None], out[2][None], point_ref
            # return torch.stack(intermediate['sgl_repr']), out[1][None], out[2][None]
            # if intermediate['coa_repr'][0] is not None:
            #     return torch.stack(intermediate['sgl_repr']), out[1][None], out[2][None], torch.stack(intermediate['coa_repr'])
            # else:
            return torch.stack(intermediate['sgl_repr'])
        
        return output[None]




class CryoformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", args=None):
        super().__init__()
        self.self_attn = FastSelfAttention(d_model, nhead)
        self.norm0 = LayerNorm(d_model)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                src,
                den_mask: Optional[Tensor] = None,
                density_pos: Optional[Tensor] = None,
                density_wei: Optional[Tensor] = None,):
        dtype = src.dtype
        q = self.with_pos_embed(src, density_pos)
        # src2 = self.self_attn(q, density_wei)
        # fastformer
        # import pdb;pdb.set_trace()
        # src2 = self.self_attn(q.transpose(0,1)).transpose(0,1)
        src2 = self.self_attn(q.transpose(0,1), den_mask).transpose(0,1)
        src2 = self.norm0(src2)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src).to(dtype)
        return src


class CryoformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", msa_embed_dim=256, pair_repr_dim=128):
        super().__init__()
        # self.args = args
        self.self_attn      = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.msa_to_single  = nn.Linear(msa_embed_dim, d_model)
        self.pair_to_single = nn.Linear(pair_repr_dim, d_model)
        self.norm_ms        = LayerNorm(d_model)
        self.norm_ps        = LayerNorm(d_model)
        self.dropout_ms     = nn.Dropout(dropout)
        self.dropout_ps     = nn.Dropout(dropout)
        self.norm0          = LayerNorm(d_model)
        
        # if args.adain:
        #     self.adain_g = nn.AdaptiveAvgPool1d(512)
        #     self.adain_d = nn.Linear(512, 1)
        #     self.adain_w = nn.Parameter(torch.randn((args.hidden_dim * 2, args.hidden_dim)))
        #     self.adain_b = nn.Parameter(torch.zeros(args.hidden_dim * 2))
        # else:
        self.multihead_attn = GaussianMultiheadAttention(d_model, nhead, dropout=dropout)
        # else:
        # self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model 
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1    = LayerNorm(d_model)
        self.norm2    = LayerNorm(d_model)
        self.norm3    = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # self.norm4    = LayerNorm(args.pair_repr_dim)
        # self.dropout4 = nn.Dropout(dropout)
        # self.outer_product_mean = OuterProductMeanSingle(
        #     input_dim=d_model, 
        #     output_dim=args.pair_repr_dim, 
        #     num_outer_channel=32, 
        #     zero_init=True)
        # if self.args.no_pair:
        
        self.nhead = nhead
        # self.layer_index = layer_index
        # if self.args.ref_point:
        #     self.smooth = nhead
        #     if layer_index == 0:
        #         self.point_ref = MLP(d_model, d_model, 3, 3)
        #         self.point_off = nn.Linear(d_model, 3 * nhead)
        #     else:
        #         self.point_off = nn.Linear(d_model, 3 * nhead)
        # if self.args.coa: 
        #     self.norm_coa      = LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    
    def forward(self, 
                tgt_sgl, tgt_msa, tgt_par, aa_embed, 
                density_repr, density_mask, den_pos, den_wei):
        """
        Args:
            tgt_sgl: N_res x B x C
            tgt_msa: N_seq x N_seq x C
            tgt_par: N_seq x N_seq x C
        """

        tgt_sgl_ms = self.msa_to_single(tgt_msa[0,:].permute(1, 0, 2))
        tgt_sgl_ms = self.norm_ms(tgt_sgl + self.dropout_ms(tgt_sgl_ms))

        tgt_sgl_ps = self.pair_to_single(tgt_par.mean(2))
        tgt_sgl_ps = self.norm_ps(tgt_sgl + self.dropout_ps(tgt_sgl_ps))

        tgt_sgl  = self.norm0(tgt_sgl_ms + tgt_sgl_ps)

        # self-attention
        q = k = self.with_pos_embed(tgt_sgl, aa_embed)
        tgt_sgl2 = self.self_attn(q, k, tgt_sgl)[0]
        tgt_sgl = tgt_sgl + self.dropout1(tgt_sgl2)
        tgt_sgl = self.norm1(tgt_sgl)
        
        
        # co-attention
        query   = self.with_pos_embed(tgt_sgl, aa_embed)
        key     = self.with_pos_embed(density_repr, den_pos)
        # import pdb;pdb.set_trace()
        
        tgt_sgl2, attn_ind = self.multihead_attn(
            query=query, key=key, value=density_repr, 
            attn_mask=None, key_padding_mask=None,
            gaussian=den_wei*8)
        # density_mask.type(torch.bool)
        # tgt_sgl2, attn_ind = self.multihead_attn(
        #     query=query, key=key, value=density_repr, 
        #     attn_mask=None, key_padding_mask=None,
        #     gaussian=den_wei*8)


        
        tgt_sgl  = self.norm2(tgt_sgl + self.dropout2(tgt_sgl2))
        tgt_sgl2 = self.linear2(self.dropout(self.activation(self.linear1(tgt_sgl))))
        tgt_sgl  = self.norm3(tgt_sgl + self.dropout3(tgt_sgl2))

        # tgt_par2 = self.outer_product_mean(tgt_sgl[:,0,:], ext_dict['seq_mask'])
        # tgt_par = self.norm4(tgt_par + self.dropout4(tgt_par2)).to(DTYPE)
        # if self.args.no_pair:
            # import pdb;pdb.set_trace()

        return tgt_sgl




class FastSelfAttention(nn.Module):
    def __init__(self, hidden_size=512,
        num_attention_heads=8,
        initializer_range= 0.02,
        pooler_type="weightpooler"):
        super(FastSelfAttention, self).__init__()
        # self.config = config
        # self.attention_head_size = attention_head_size
        self.num_attention_heads = num_attention_heads
        self.initializer_range = initializer_range

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" %
                (hidden_size, num_attention_heads))
        self.attention_head_size = int(hidden_size /num_attention_heads)
        self.num_attention_heads = num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.input_dim= hidden_size
        
        self.query      = nn.Linear(self.input_dim, self.all_head_size)
        self.query_att  = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.key        = nn.Linear(self.input_dim, self.all_head_size)
        self.key_att    = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.transform  = nn.Linear(self.all_head_size, self.all_head_size)

        # self.softmax = nn.Softmax(dim=-1)
        
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
                
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        # batch_size, seq_len, num_head * head_dim, batch_size, seq_len
        # print("hidden_states:", hidden_states.shape)
        batch_size, seq_len, _ = hidden_states.shape
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        # batch_size, num_head, seq_len
        query_for_score = self.query_att(mixed_query_layer).transpose(1, 2) / self.attention_head_size**0.5
        # add attention mask
        if attention_mask is not None:
            query_for_score += attention_mask.float().unsqueeze(1) 

        # batch_size, num_head, 1, seq_len
        # query_weight = F.softmax(query_for_score, -1, dtype=self.dtype).unsqueeze(2)
        query_weight = F.softmax(query_for_score, -1).unsqueeze(2)

        # batch_size, num_head, seq_len, head_dim
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # batch_size, num_head, head_dim, 1
        pooled_query = torch.matmul(query_weight, query_layer).transpose(1, 2).view(-1,1,self.num_attention_heads*self.attention_head_size)
        pooled_query_repeat= pooled_query.repeat(1, seq_len,1)
        # batch_size, num_head, seq_len, head_dim

        # batch_size, num_head, seq_len
        mixed_query_key_layer=mixed_key_layer* pooled_query_repeat
        
        query_key_score=(self.key_att(mixed_query_key_layer)/ self.attention_head_size**0.5).transpose(1, 2)
        
        # add attention mask
        if attention_mask is not None:
            query_key_score +=attention_mask.unsqueeze(1)

        # batch_size, num_head, 1, seq_len
        # query_key_weight = F.softmax(query_key_score, -1, dtype=self.dtype).unsqueeze(2)
        query_key_weight = F.softmax(query_key_score, -1).unsqueeze(2)

        key_layer = self.transpose_for_scores(mixed_query_key_layer)
        pooled_key = torch.matmul(query_key_weight, key_layer)

        #query = value
        weighted_value =(pooled_key * query_layer).transpose(1, 2)
        weighted_value = weighted_value.reshape(
            weighted_value.size()[:-2] + (self.num_attention_heads * self.attention_head_size,))
        weighted_value = self.transform(weighted_value) + mixed_query_layer
      
        return weighted_value


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

