
from tkinter import E
import torch
import torch.nn as nn
from typing import Tuple, Optional

import pytorch_lightning as pl
from cryofold.model.primitives import Linear, LayerNorm
from cryofold.utils.tensor_utils import add, one_hot


class RelPositionalEmbedding(nn.Module):
    def __init__(self, hidden_size=384):
        super(RelPositionalEmbedding, self).__init__()

        self.hidden_size = hidden_size

        inv_freq = 1 / (10000 ** (torch.arange(0.0, hidden_size, 2.0) / hidden_size))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq):
        # import pdb;pdb.set_trace()
        has_batch=True
        if len(pos_seq.shape)==1:
            pos_seq = pos_seq[None,...]
            has_batch=False
        B, N_res = pos_seq.shape
        pos_emb = torch.zeros((B, N_res, self.hidden_size), device=pos_seq.device, dtype=pos_seq.dtype)

        pos_emb[:, :, 0::2] = torch.sin(pos_seq.unsqueeze(-1) * self.inv_freq)
        pos_emb[:, :, 1::2] = torch.cos(pos_seq.unsqueeze(-1) * self.inv_freq)
        if has_batch:
            return pos_emb
        else:
            return pos_emb[0]

class InputEmbedder(pl.LightningModule):
    """
    Embeds a subset of the input features.

    Implements Algorithms 3 (InputEmbedder) and 4 (relpos).
    """

    def __init__(
        self,
        tf_dim: int,
        msa_dim: int,
        c_z: int,
        c_m: int,
        c_s: int,
        relpos_k: int,
        use_single: bool = False,
        use_chain: bool = False,
        **kwargs,
    ):
        """
        Args:
            tf_dim:
                Final dimension of the target features
            msa_dim:
                Final dimension of the MSA features
            c_z:
                Pair embedding dimension
            c_m:
                MSA embedding dimension
            relpos_k:
                Window size used in relative positional encoding
        """
        super(InputEmbedder, self).__init__()

        self.tf_dim = tf_dim
        self.msa_dim = msa_dim

        self.c_z = c_z
        self.c_m = c_m
        self.c_s = c_s
        self.use_single = use_single
        self.use_chain = use_chain

        self.linear_tf_z_i = Linear(tf_dim, c_z)
        self.linear_tf_z_j = Linear(tf_dim, c_z)
        self.linear_tf_m   = Linear(tf_dim, c_m)
        self.linear_msa_m  = Linear(msa_dim, c_m)

        # RPE stuff
        self.relpos_k = relpos_k
        self.no_bins = 2 * relpos_k + 1
        self.linear_relpos = Linear(self.no_bins, c_z)

        if use_single:
            self.linear_embed_s  = Linear(tf_dim, c_s)
            self.relpos_enc      = RelPositionalEmbedding(hidden_size=c_s)
        if use_chain:
            self.linear_embed_c  = Linear(tf_dim, c_s)
            self.chaidx_enc      = RelPositionalEmbedding(hidden_size=c_s)

            # self.linear_relpos_s = Linear(self.no_bins, c_s)


    def relpos(self, ri: torch.Tensor):
        """
        Computes relative positional encodings

        Implements Algorithm 4.

        Args:
            ri:
                "residue_index" features of shape [*, N]
        """
        d = ri[..., None] - ri[..., None, :]
        boundaries = torch.arange(
            start=-self.relpos_k, end=self.relpos_k + 1, device=d.device
        ) 
        reshaped_bins = boundaries.view(((1,) * len(d.shape)) + (len(boundaries),))
        d = d[..., None] - reshaped_bins
        d = torch.abs(d)
        d = torch.argmin(d, dim=-1)
        d = nn.functional.one_hot(d, num_classes=len(boundaries)).float()
        d = d.to(ri.dtype)
        return self.linear_relpos(d)
    

    def forward(
        self,
        tf: torch.Tensor,
        ri: torch.Tensor,
        msa: torch.Tensor,
        inplace_safe: bool = False,
        ci: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tf:
                "target_feat" features of shape [*, N_res, tf_dim]
            ri:
                "residue_index" features of shape [*, N_res]
            msa:
                "msa_feat" features of shape [*, N_clust, N_res, msa_dim]
        Returns:
            msa_emb:
                [*, N_clust, N_res, C_m] MSA embedding
            pair_emb:
                [*, N_res, N_res, C_z] pair embedding

        """
        # [*, N_res, c_z]
        tf_emb_i = self.linear_tf_z_i(tf)
        tf_emb_j = self.linear_tf_z_j(tf)

        if self.use_single:
            sgl_pos_emb = self.relpos_enc(ri.type(tf_emb_i.dtype))
            # [*, N_res, c_s]
            tf_new = torch.cat([tf[...,1:], tf[...,:1]],dim=-1)
            sgl_emb = self.linear_embed_s(tf_new) + sgl_pos_emb
        else:
            sgl_emb = None

        if self.use_chain:
            cha_pos_emb = self.chaidx_enc(ci.type(tf_emb_i.dtype))
            # [*, N_res, c_s]
            tf_new = torch.cat([tf[...,1:], tf[...,:1]],dim=-1)
            cha_emb = self.linear_embed_c(tf_new) + cha_pos_emb
        else:
            cha_emb = None

        # [*, N_res, N_res, c_z]
        pair_emb = self.relpos(ri.type(tf_emb_i.dtype))
        pair_emb = add(pair_emb, 
            tf_emb_i[..., None, :], 
            inplace=inplace_safe
        )
        pair_emb = add(pair_emb, 
            tf_emb_j[..., None, :, :], 
            inplace=inplace_safe
        )

        # [*, N_clust, N_res, c_m]
        n_clust = msa.shape[-3]
        tf_m = (
            self.linear_tf_m(tf)
            .unsqueeze(-3)
            .expand(((-1,) * len(tf.shape[:-2]) + (n_clust, -1, -1)))
        )
        msa_emb = self.linear_msa_m(msa) + tf_m

        return cha_emb, sgl_emb, msa_emb, pair_emb

class RecyclingEmbedder(pl.LightningModule):
    """
    Embeds the output of an iteration of the model for recycling.

    Implements Algorithm 32.
    """
    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_s: int,
        min_bin: float,
        max_bin: float,
        no_bins: int,
        inf: float = 1e8,
        use_single: bool = False,
        use_chain: bool = False,
        **kwargs,
    ):
        """
        Args:
            c_m:
                MSA channel dimension
            c_z:
                Pair embedding channel dimension
            min_bin:
                Smallest distogram bin (Angstroms)
            max_bin:
                Largest distogram bin (Angstroms)
            no_bins:
                Number of distogram bins
        """
        super(RecyclingEmbedder, self).__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.c_s = c_s
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.inf = inf
        self.use_single = use_single
        self.use_chain  = use_chain

        self.linear = Linear(self.no_bins, self.c_z)
        self.layer_norm_m = LayerNorm(self.c_m)
        self.layer_norm_z = LayerNorm(self.c_z)
        if self.use_single:
            self.layer_norm_s = LayerNorm(self.c_s)
        if self.use_chain:
            self.layer_norm_c = LayerNorm(self.c_s)


    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        x: torch.Tensor,
        s: torch.Tensor = None,
        c: torch.Tensor = None,
        inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m:
                First row of the MSA embedding. [*, N_res, C_m]
            z:
                [*, N_res, N_res, C_z] pair embedding
            x:
                [*, N_res, 3] predicted C_beta coordinates
            s:
                [*, N_res, c_s] Single embedding
        Returns:
            s:
                [*, N_res, C_s] Single embedding update
            m:
                [*, N_res, C_m] MSA embedding update
            z:
                [*, N_res, N_res, C_z] pair embedding update
        """
        if self.use_single:
            assert s is not None
            # [*, N, C_s]
            s_update = self.layer_norm_s(s)
            if(inplace_safe):
                s.copy_(s_update)
                s_update = s
        else:
            s_update = None

        if self.use_chain:
            assert c is not None
            # [*, N, C_s]
            c_update = self.layer_norm_c(c)
            if(inplace_safe):
                c.copy_(c_update)
                c_update = c
        else:
            c_update = None

        # [*, N, C_m]
        m_update = self.layer_norm_m(m)
        if(inplace_safe):
            m.copy_(m_update)
            m_update = m

        # [*, N, N, C_z]
        z_update = self.layer_norm_z(z)
        if(inplace_safe):
            z.copy_(z_update)
            z_update = z

        # This squared method might become problematic in FP16 mode.
        bins = torch.linspace(
            self.min_bin,
            self.max_bin,
            self.no_bins,
            dtype=x.dtype,
            device=x.device,
            requires_grad=False,
        )
        squared_bins = bins ** 2
        upper = torch.cat(
            [squared_bins[1:], squared_bins.new_tensor([self.inf])], dim=-1
        )
        d = torch.sum(
            (x[..., None, :] - x[..., None, :, :]) ** 2, dim=-1, keepdims=True
        )

        # [*, N, N, no_bins]
        d = ((d > squared_bins) * (d < upper)).type(x.dtype)

        # [*, N, N, C_z]
        d = self.linear(d)
        z_update = add(z_update, d, inplace_safe)
        
        return c_update, s_update, m_update, z_update
        # if self.use_single:
        #     if self.use_chain:
        #         return c_update, s_update, m_update, z_update
        #     else:
        #         return s_update, m_update, z_update
        # else:
        #     return m_update, z_update


class TemplateAngleEmbedder(pl.LightningModule):
    """
    Embeds the "template_angle_feat" feature.

    Implements Algorithm 2, line 7.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        **kwargs,
    ):
        """
        Args:
            c_in:
                Final dimension of "template_angle_feat"
            c_out:
                Output channel dimension
        """
        super(TemplateAngleEmbedder, self).__init__()

        self.c_out = c_out
        self.c_in = c_in

        self.linear_1 = Linear(self.c_in, self.c_out, init="relu")
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.c_out, self.c_out, init="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [*, N_templ, N_res, c_in] "template_angle_feat" features
        Returns:
            x: [*, N_templ, N_res, C_out] embedding
        """
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)

        return x


class TemplatePairEmbedder(pl.LightningModule):
    """
    Embeds "template_pair_feat" features.

    Implements Algorithm 2, line 9.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        **kwargs,
    ):
        """
        Args:
            c_in:

            c_out:
                Output channel dimension
        """
        super(TemplatePairEmbedder, self).__init__()

        self.c_in = c_in
        self.c_out = c_out

        # Despite there being no relu nearby, the source uses that initializer
        self.linear = Linear(self.c_in, self.c_out, init="relu")

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, C_in] input tensor
        Returns:
            [*, C_out] output tensor
        """
        x = self.linear(x)

        return x


class ExtraMSAEmbedder(pl.LightningModule):
    """
    Embeds unclustered MSA sequences.

    Implements Algorithm 2, line 15
    """
    def __init__(
        self,
        c_in: int,
        c_out: int,
        **kwargs,
    ):
        """
        Args:
            c_in:
                Input channel dimension
            c_out:
                Output channel dimension
        """
        super(ExtraMSAEmbedder, self).__init__()

        self.c_in = c_in
        self.c_out = c_out

        self.linear = Linear(self.c_in, self.c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_extra_seq, N_res, C_in] "extra_msa_feat" features
        Returns:
            [*, N_extra_seq, N_res, C_out] embedding
        """
        x = self.linear(x)

        return x
