
from functools import partial
import weakref

import torch
import torch.nn as nn

import pytorch_lightning as pl

from cryofold.model.primitives import LayerNorm

from cryofold.model.embedders import (
    InputEmbedder,
    RecyclingEmbedder,
    TemplateAngleEmbedder,
    TemplatePairEmbedder,
    ExtraMSAEmbedder,
)
from cryofold.model.evoformer import EvoformerStack, ExtraMSAStack
from cryofold.model.heads import AuxiliaryHeads
from cryofold.model.structure_module import StructureModule
from cryofold.model.template import (
    TemplatePairStack,
    TemplatePointwiseAttention,
    embed_templates_average,
    embed_templates_offload,
)
import cryofold.np.residue_constants as residue_constants
from cryofold.utils.feats import (
    pseudo_beta_fn,
    build_extra_msa_feat,
    build_template_angle_feat,
    build_template_pair_feat,
    atom14_to_atom37,
)

from cryofold.utils.tensor_utils import (
    add,
    dict_multimap,
    tensor_tree_map,
)

from cryofold.utils.protein_utils import save_dict_as_pdb
from cryofold.utils.rigid_utils import to_backbone_frame, to_normed_ca


class CryoFold(pl.LightningModule):
    """
    Alphafold 2.

    Implements Algorithm 2 (but with training).
    """

    def __init__(self, config):
        """
        Args:
            config:
                A dict-like config object (like the one in config.py)
        """
        super(CryoFold, self).__init__()

        self.globals = config.globals
        self.config = config.model
        self.cryoem_config = self.config.cryoem
        self.template_config = self.config.template
        self.extra_msa_config = self.config.extra_msa
        self.sm_config = self.config.structure_module

        # Main trunk + structure module
        self.input_embedder = InputEmbedder(
            **self.config["input_embedder"],
        )
        self.recycling_embedder = RecyclingEmbedder(
            **self.config["recycling_embedder"],
        )

        if(self.cryoem_config.enabled):
            from cryofold.model.cryoem_embedder import (
                    CryoformerEmbedder,
                    CryoformerEncoder,
                    CryoformerDecoder,
                )
            self.cryoformer_emb = CryoformerEmbedder(
                self.cryoem_config["cryoformer_embedder"],
            )
            self.cryoformer_enc = CryoformerEncoder(
                **self.cryoem_config["cryoformer_encoder"],
            )
            self.cryoformer_dec = CryoformerDecoder(
                **self.cryoem_config["cryoformer_decoder"],
            )
            self.cryo_sgl_norm = LayerNorm(self.cryoem_config["cryoformer_embedder"]['hidden_dim'])
        
        if(self.template_config.enabled):
            self.template_angle_embedder = TemplateAngleEmbedder(
                **self.template_config["template_angle_embedder"],
            )
            self.template_pair_embedder = TemplatePairEmbedder(
                **self.template_config["template_pair_embedder"],
            )
            self.template_pair_stack = TemplatePairStack(
                **self.template_config["template_pair_stack"],
            )
            self.template_pointwise_att = TemplatePointwiseAttention(
                **self.template_config["template_pointwise_attention"],
            )
       
        if(self.extra_msa_config.enabled):
            self.extra_msa_embedder = ExtraMSAEmbedder(
                **self.extra_msa_config["extra_msa_embedder"],
            )
            self.extra_msa_stack = ExtraMSAStack(
                **self.extra_msa_config["extra_msa_stack"],
            )
        
        self.evoformer = EvoformerStack(
            **self.config["evoformer_stack"],
        )

        if(self.config.backbone_frame.enabled):
            from cryofold.model.heads import BackboneFrameHead
            self.backbone_frame = BackboneFrameHead(
                **self.config["backbone_frame"],
            )
        
        # if self.config.heads.hungarian_matcher.enabled:
        #     from cryofold.model.heads import HungarianMatcherHead
        #     self.matcher = HungarianMatcherHead(
        #         **self.config.heads["hungarian_matcher"],
        #     )
        if(self.sm_config.enabled):
            self.structure_module = StructureModule(
                **self.config["structure_module"],
            )
        self.aux_heads = AuxiliaryHeads(
            self.config["heads"],
        )

    def embed_templates(self, batch, z, pair_mask, templ_dim, inplace_safe): 
        if(self.template_config.offload_templates):
            return embed_templates_offload(self, 
                batch, z, pair_mask, templ_dim, inplace_safe=inplace_safe,
            )
        elif(self.template_config.average_templates):
            return embed_templates_average(self, 
                batch, z, pair_mask, templ_dim, inplace_safe=inplace_safe,
            )

        # Embed the templates one at a time (with a poor man's vmap)
        pair_embeds = []
        n = z.shape[-2]
        n_templ = batch["template_aatype"].shape[templ_dim]
        # import pdb;pdb.set_trace()
        if(inplace_safe):
            # We'll preallocate the full pair tensor now to avoid manifesting
            # a second copy during the stack later on
            t_pair = z.new_zeros(
                z.shape[:-3] + 
                (n_templ, n, n, self.globals.c_t)
            )

        for i in range(n_templ):
            idx = batch["template_aatype"].new_tensor(i)
            single_template_feats = tensor_tree_map(
                lambda t: torch.index_select(t, templ_dim, idx).squeeze(templ_dim),
                batch,
            )

            # [*, N, N, C_t]
            t = build_template_pair_feat(
                single_template_feats,
                use_unit_vector=self.config.template.use_unit_vector,
                inf=self.config.template.inf,
                eps=self.config.template.eps,
                **self.config.template.distogram,
            ).to(z.dtype)
            t = self.template_pair_embedder(t)

            if(inplace_safe):
                t_pair[..., i, :, :, :] = t
            else:
                pair_embeds.append(t)
            
            del t

        if(not inplace_safe):
            t_pair = torch.stack(pair_embeds, dim=templ_dim)
       
        del pair_embeds

        # [*, S_t, N, N, C_z]
        t = self.template_pair_stack(
            t_pair, 
            pair_mask.unsqueeze(-3).to(dtype=z.dtype), 
            chunk_size=self.globals.chunk_size,
            use_lma=self.globals.use_lma,
            inplace_safe=inplace_safe,
            _mask_trans=self.config._mask_trans,
        )
        del t_pair

        # [*, N, N, C_z]
        t = self.template_pointwise_att(
            t, 
            z, 
            template_mask=batch["template_mask"].to(dtype=z.dtype),
            use_lma=self.globals.use_lma,
        )
        t_mask = torch.sum(batch["template_mask"], dim=-1) > 0
        # Append singletons
        t_mask = t_mask.reshape(
            *t_mask.shape, *([1] * (len(t.shape) - len(t_mask.shape)))
        )

        if(inplace_safe):
            t *= t_mask
        else:
            # t = t * (torch.sum(batch["template_mask"], dim=-1) > 0).view(-1,1,1,1)
            t = t * t_mask

        ret = {}

        ret.update({"template_pair_embedding": t})

        del t

        if self.config.template.embed_angles:
            template_angle_feat = build_template_angle_feat(
                batch
            )

            # [*, S_t, N, C_m]
            a = self.template_angle_embedder(template_angle_feat)

            ret["template_angle_embedding"] = a

        return ret

    def prepare_recycling_inputs(self, prevs, batch_dims, n, feats, m, z):
        # Unpack the recycling embeddings. Removing them from the list allows 
        # them to be freed further down in this function, saving memory
        m_1_prev, z_prev, x_prev, s_prev = reversed([prevs.pop() for _ in range(4)])
        # import pdb;pdb.set_trace()
        # Initialize the recycling embeddings, if needs be 
        if None in [m_1_prev, z_prev, x_prev, s_prev]:
            # [*, N, C_m]
            m_1_prev = m.new_zeros(
                (*batch_dims, n, self.config.input_embedder.c_m),
                requires_grad=False,
            )

            # [*, N, N, C_z]
            z_prev = z.new_zeros(
                (*batch_dims, n, n, self.config.input_embedder.c_z),
                requires_grad=False,
            )

            # [*, N, 3]
            x_prev = z.new_zeros(
                (*batch_dims, n, residue_constants.atom_type_num, 3),
                requires_grad=False,
            )

            # [*, N, C_m]
            s_prev = m.new_zeros(
                (*batch_dims, n, self.config.input_embedder.c_s),
                requires_grad=False,
            )

        if(self.globals.use_chain):
            c_prev = m.new_zeros(
                (*batch_dims, n, self.config.input_embedder.c_s),
                requires_grad=False,
            )
        else:
            c_prev = None

        x_prev = pseudo_beta_fn(
            feats["aatype"], x_prev, None
        ).to(dtype=z.dtype)

        return m_1_prev, z_prev, x_prev, s_prev, c_prev

    def iteration(self, feats, prevs, _recycle=True):
        # Primary output dictionary
        outputs = {}

        # cryoem density embedding and cryoformer encoder
        """
        density: [1, 104, 80, 120]
        density_mask: [1, 104, 80, 120]
        density_whl: [1, 3]
        density_offset: [1, 3]
        density_cropidx: [1, 3]
        """
        dtype = next(self.parameters()).dtype
        # feats['dtype'] = dtype
        # This needs to be done manually for DeepSpeed's sake
        for k in feats:
            if(feats[k].dtype == torch.float32) or (feats[k].dtype == torch.float16) or (feats[k].dtype == torch.float64):
                feats[k] = feats[k].to(dtype=dtype)

        if(self.cryoem_config.enabled):
            # cryoformer
            # import pdb;pdb.set_trace()
            cryo_repr = self.cryoformer_emb(feats)
            den_b, den_c, den_w, den_h, den_l = cryo_repr['density_repr'].shape


            density_repr = cryo_repr['density_repr'].flatten(2).permute(2, 0, 1)
            density_mask = cryo_repr['density_mask'].flatten(1)#.permute(1, 0)
            density_pos  = cryo_repr['density_pos'].flatten(2).permute(2, 0, 1)
            density_wei  = cryo_repr['density_wei'].flatten(1)#.permute(1, 0)

            # density_mask = None
            # import pdb;pdb.set_trace()
            # pdb.set_trace=lambda:None

            cryo_repr_self = self.cryoformer_enc(density_repr, density_mask, density_pos, density_wei)
            outputs["cryoem_density"] = feats['cryoem_density'] 
            outputs["cryoem_size"]    = feats['cryoem_size']
            outputs['density_repr']   = cryo_repr_self.permute(1,2,0).view(den_b, den_c, den_w, den_h, den_l)
            outputs["density_feats"]  = cryo_repr["density_feats"] 
            # print("no embedder")
            del cryo_repr

        if torch.isnan(cryo_repr_self).any():
            import pdb;pdb.set_trace()
        # Grab some data about the input
        batch_dims = feats["target_feat"].shape[:-2]
        no_batch_dims = len(batch_dims)
        n = feats["target_feat"].shape[-2]
        n_seq = feats["msa_feat"].shape[-3]
        device = feats["target_feat"].device
        
        # Controls whether the model uses in-place operations throughout
        # The dual condition accounts for activation checkpoints
        inplace_safe = not (self.training or torch.is_grad_enabled())

        # Prep some features
        seq_mask  = feats["seq_mask"]
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        msa_mask  = feats["msa_mask"]
        
        ## Initialize the MSA and pair representations
        c, s, m, z = self.input_embedder(
            feats["target_feat"],
            feats["residue_index"],
            feats["msa_feat"],
            inplace_safe=inplace_safe,
            ci=feats["chain_index"],
        )
        

        m_1_prev, z_prev, x_prev, s_prev, c_prev = self.prepare_recycling_inputs(
            prevs, batch_dims, n, feats, m, z)
        

        # The recycling embedder is memory-intensive, so we offload first
        if(self.globals.offload_inference and inplace_safe):
            m = m.cpu()
            z = z.cpu()
        c_prev_emb, s_prev_emb, m_1_prev_emb, z_prev_emb = self.recycling_embedder(
            m_1_prev,
            z_prev,
            x_prev,
            s_prev,
            c_prev,
            inplace_safe=inplace_safe,
        )

        if(self.globals.offload_inference and inplace_safe):
            m = m.to(m_1_prev_emb.device)
            z = z.to(z_prev.device)


        # [*, S_c, N, C_m]
        m[..., 0, :, :] += m_1_prev_emb

        # [*, N, N, C_z]
        z = add(z, z_prev_emb, inplace=inplace_safe)

        # [*, N, C_s]
        if(self.globals.use_single):
            s += s_prev_emb

        # [*, N, C_s]
        if(self.globals.use_chain):
            c += c_prev_emb


        # Deletions like these become significant for inference with large N,
        # where they free unused tensors and remove references to others such
        # that they can be offloaded later
        del m_1_prev, z_prev, x_prev, m_1_prev_emb, z_prev_emb

        # Embed the templates + merge with MSA/pair embeddings
        if self.config.template.enabled: 
            template_feats = {
                k: v for k, v in feats.items() if k.startswith("template_")
            }
            # print("1 template_mask: ", template_feats['template_mask'])
            # import pdb;pdb.set_trace()
            template_embeds = self.embed_templates(
                template_feats,
                z,
                pair_mask.to(dtype=z.dtype),
                no_batch_dims,
                inplace_safe=inplace_safe,
            )
            # import pdb;pdb.set_trace()

            # [*, N, N, C_z]
            z = add(z,
                template_embeds.pop("template_pair_embedding"),
                inplace_safe,
            )
            # import pdb;pdb.set_trace()


            if "template_angle_embedding" in template_embeds:
                # [*, S = S_c + S_t, N, C_m]
                m = torch.cat(
                    [m, template_embeds["template_angle_embedding"]], 
                    dim=-3
                )

                # [*, S, N]
                torsion_angles_mask = feats["template_torsion_angles_mask"]
                msa_mask = torch.cat(
                    [feats["msa_mask"], torsion_angles_mask[..., 2]], 
                    dim=-2
                )

        # Embed extra MSA features + merge with pairwise embeddings
        if self.config.extra_msa.enabled:
            # [*, S_e, N, C_e]
            a = self.extra_msa_embedder(build_extra_msa_feat(feats))

            if(self.globals.offload_inference):
                # To allow the extra MSA stack (and later the evoformer) to
                # offload its inputs, we remove all references to them here
                input_tensors = [a, z]
                del a, z
                # [*, N, N, C_z]
                z = self.extra_msa_stack._forward_offload(
                    input_tensors,
                    msa_mask=feats["extra_msa_mask"].to(dtype=m.dtype),
                    chunk_size=self.globals.chunk_size,
                    use_lma=self.globals.use_lma,
                    pair_mask=pair_mask.to(dtype=m.dtype),
                    _mask_trans=self.config._mask_trans,
                )
    
                del input_tensors
            else:
                # import pdb;pdb.set_trace()

                # [*, N, N, C_z]
                z = self.extra_msa_stack(
                    a, z,
                    msa_mask=feats["extra_msa_mask"].to(dtype=m.dtype),
                    chunk_size=self.globals.chunk_size,
                    use_lma=self.globals.use_lma,
                    pair_mask=pair_mask.to(dtype=m.dtype),
                    inplace_safe=inplace_safe,
                    _mask_trans=self.config._mask_trans,
                )

        # Run MSA + pair embeddings through the trunk of the network
        # m: [*, S, N, C_m]
        # z: [*, N, N, C_z]
        # s: [*, N, C_s]  
        if torch.isnan(m).any():
            print("msa has nan.")
            import pdb;pdb.set_trace()

        if self.config.evoformer_stack.freeze:
            with torch.no_grad(): 
                if(self.globals.offload_inference):
                    input_tensors = [m, z]
                    del m, z
                    m, z, s_m1 = self.evoformer._forward_offload(
                        input_tensors,
                        msa_mask=msa_mask.to(dtype=input_tensors[0].dtype),
                        pair_mask=pair_mask.to(dtype=input_tensors[1].dtype),
                        chunk_size=self.globals.chunk_size,
                        use_lma=self.globals.use_lma,
                        use_flash=self.globals.use_flash,
                        _mask_trans=self.config._mask_trans,
                    )
            
                    del input_tensors
                else:
                    m, z, s_m1 = self.evoformer(
                        m,
                        z,
                        msa_mask=msa_mask.to(dtype=m.dtype),
                        pair_mask=pair_mask.to(dtype=z.dtype),
                        chunk_size=self.globals.chunk_size,
                        use_lma=self.globals.use_lma,
                        use_flash=self.globals.use_flash,
                        inplace_safe=inplace_safe,
                        _mask_trans=self.config._mask_trans,
                    )
        else:
            if(self.globals.offload_inference):
                input_tensors = [m, z]
                del m, z
                m, z, s_m1 = self.evoformer._forward_offload(
                    input_tensors,
                    msa_mask=msa_mask.to(dtype=input_tensors[0].dtype),
                    pair_mask=pair_mask.to(dtype=input_tensors[1].dtype),
                    chunk_size=self.globals.chunk_size,
                    use_lma=self.globals.use_lma,
                    use_flash=self.globals.use_flash,
                    _mask_trans=self.config._mask_trans,
                )
        
                del input_tensors
            else:
                m, z, s_m1 = self.evoformer(
                    m,
                    z,
                    msa_mask=msa_mask.to(dtype=m.dtype),
                    pair_mask=pair_mask.to(dtype=z.dtype),
                    chunk_size=self.globals.chunk_size,
                    use_lma=self.globals.use_lma,
                    use_flash=self.globals.use_flash,
                    inplace_safe=inplace_safe,
                    _mask_trans=self.config._mask_trans,
                )


        outputs["msa"] = m[..., :n_seq, :, :]
        outputs["pair"] = z
        outputs["single"] = s_m1
        if torch.isnan(s_m1).any():
            print("single has nan.")
            import pdb;pdb.set_trace()

        del z

        if(self.cryoem_config.enabled):
            bf_single = self.cryo_sgl_norm(outputs["single"]) 
            # use aa embedding as single 
            if(self.globals.use_single):
                bf_single = bf_single + s
                
            # use chain embedding  
            if(self.globals.use_chain):
                bf_single = bf_single + c

            # cryoformer
            has_batch = True
            if len(bf_single.shape)==2: 
                tgt_sgl = bf_single.unsqueeze(1)
                tgt_msa = outputs['msa'].unsqueeze(1)
                tgt_par = outputs['pair'].unsqueeze(1)
                # s_prev  = s_prev.unsqueeze(1)
                has_batch = False
            else:
                # import pdb;pdb.set_trace()   
                # [384, 1, 384]
                tgt_sgl = bf_single.permute(1, 0, 2)
                # [512, 1, 384, 256]
                tgt_msa = outputs['msa'].permute(1, 0, 2, 3)
                # [384, 1, 384, 128]
                tgt_par = outputs['pair'].permute(1, 0, 2 ,3)
                # s_prev  = s_prev.permute(1, 0, 2)

            # import pdb;pdb.set_trace()    
            decoder_param = [
                tgt_sgl, tgt_msa, tgt_par, tgt_sgl,
                cryo_repr_self, density_mask, density_pos, density_wei
            ]
            out = self.cryoformer_dec(*decoder_param)
            out = out.transpose(1, 2)
            # import pdb;pdb.set_trace()
            outputs["single_dec"] = out
            if torch.isnan(out).any():
                print("cryoformer_dec has nan.")
                import pdb;pdb.set_trace()

            if has_batch:
                if(self.globals.fuse_single):
                    outputs["single"] = out[-1]
            else:
                if(self.globals.fuse_single):
                    outputs["single"] = out[-1,0]
                # outputs["single_dec"] = out[:,0]
            del cryo_repr_self, density_mask, density_pos, density_wei
            

            # out = outputs["single"]
            # out = out[None,:,:,:]

        # import pdb;pdb.set_trace()
        if self.cryoem_config.enabled and self.config.backbone_frame.enabled:
            # backbone frame in sigmoid space
            normed_backbone_frame = self.backbone_frame(
                outputs["single_dec"]
            )
            outputs[
                "normed_backbone_frame"
            ] = normed_backbone_frame
            # backbone frame in protein/physics space
            backbone_frame = to_backbone_frame(
                normed_backbone_frame, feats
            ) 
            # import pdb;pdb.set_trace()

            if has_batch:
                outputs['backbone_frame'] = backbone_frame
                outputs["final_atom_positions"] = backbone_frame[...,4:][:,:,:,None,:].repeat([1,1,1,37,1])
                outputs["final_atom_mask"] = feats["atom37_atom_exists"][None,:,:]
            else:
                outputs['backbone_frame'] = backbone_frame[:,0]
                outputs["final_atom_positions"] = backbone_frame[:,0,:,4:][:,:,None,:].repeat([1,1,37,1])
                outputs["final_atom_mask"] = feats["atom37_atom_exists"]


        # import pdb;pdb.set_trace()


        # Predict 3D structure
        if(self.sm_config.enabled):
            outputs["sm"] = self.structure_module(
                outputs,
                feats["aatype"],
                mask=feats["seq_mask"].to(dtype=m.dtype),
                inplace_safe=inplace_safe,
                _offload_inference=self.globals.offload_inference,
            )
            N_aux = outputs["sm"]["positions"].shape[0]
            poss = []
            for i in range(N_aux):
                pos = atom14_to_atom37(
                    outputs["sm"]["positions"][i], feats
                )
                poss.append(pos)
            outputs["final_atom_positions"] = torch.stack(poss)
            outputs["final_atom_mask"] = feats["atom37_atom_exists"]
            # outputs["final_affinex_tensor"] = outputs["sm"]["frames"][-1]

        # import pdb;pdb.set_trace()
        if self.sm_config.enabled and self.cryoem_config.enabled and self.config.backbone_frame.enabled:
            if has_batch:
                ca_positions = outputs["final_atom_positions"][:,:,:,1,:]
            else:
                ca_positions = outputs["final_atom_positions"][:,:,1,:]
            
            normed_ca_positions = to_normed_ca(
                ca_positions, feats
            )
            outputs["final_normed_ca_positions"] = normed_ca_positions

        # if self.config.heads.hungarian_matcher.enabled:
        #     match_indices = self.matcher(
        #         outputs, feats
        #     )
        #     import pdb;pdb.set_trace()
        #     outputs["match_indices"] = match_indices
        # Save embeddings for use during the next recycling iteration

        # [*, N, C_m]
        m_1_prev = m[..., 0, :, :]

        # [*, N, C_m]
        s_prev = outputs["single"]
        # import pdb;pdb.set_trace()

        # [*, N, N, C_z]
        z_prev = outputs["pair"]

        # [*, N, 3]
        x_prev = outputs["final_atom_positions"][-1]


        return outputs, m_1_prev, z_prev, x_prev, s_prev

    def forward(self, batch, global_step=-1):
        """
        Args:
            batch:
                Dictionary of arguments outlined in Algorithm 2. Keys must
                include the official names of the features in the
                supplement subsection 1.2.9.

                The final dimension of each input must have length equal to
                the number of recycling iterations.

                Features (without the recycling dimension):

                    "aatype" ([*, N_res]):
                        Contrary to the supplement, this tensor of residue
                        indices is not one-hot.
                    "target_feat" ([*, N_res, C_tf])
                        One-hot encoding of the target sequence. C_tf is
                        config.model.input_embedder.tf_dim.
                    "residue_index" ([*, N_res])
                        Tensor whose final dimension consists of
                        consecutive indices from 0 to N_res.
                    "msa_feat" ([*, N_seq, N_res, C_msa])
                        MSA features, constructed as in the supplement.
                        C_msa is config.model.input_embedder.msa_dim.
                    "seq_mask" ([*, N_res])
                        1-D sequence mask
                    "msa_mask" ([*, N_seq, N_res])
                        MSA mask
                    "pair_mask" ([*, N_res, N_res])
                        2-D pair mask
                    "extra_msa_mask" ([*, N_extra, N_res])
                        Extra MSA mask
                    "template_mask" ([*, N_templ])
                        Template mask (on the level of templates, not
                        residues)
                    "template_aatype" ([*, N_templ, N_res])
                        Tensor of template residue indices (indices greater
                        than 19 are clamped to 20 (Unknown))
                    "template_all_atom_positions"
                        ([*, N_templ, N_res, 37, 3])
                        Template atom coordinates in atom37 format
                    "template_all_atom_mask" ([*, N_templ, N_res, 37])
                        Template atom coordinate mask
                    "template_pseudo_beta" ([*, N_templ, N_res, 3])
                        Positions of template carbon "pseudo-beta" atoms
                        (i.e. C_beta for all residues but glycine, for
                        for which C_alpha is used instead)
                    "template_pseudo_beta_mask" ([*, N_templ, N_res])
                        Pseudo-beta mask
        """

        # Initialize recycling embeddings
        m_1_prev, z_prev, x_prev, s_prev = None, None, None, None
        prevs = [m_1_prev, z_prev, x_prev, s_prev]

        is_grad_enabled = torch.is_grad_enabled()
        # is_grad_enabled =True
        """
        density: [1, 104, 80, 120, 1]
        density_mask: [1, 104, 80, 120, 1]
        density_whl: [1, 3, 1]
        density_offset: [1, 3, 1]
        density_cropidx: [1, 3, 1]
        """
        # import pdb;pdb.set_trace()
        # Main recycling loop
        num_iters = batch["aatype"].shape[-1]
        # print("num_iters:", num_iters)
        for cycle_no in range(num_iters): 
            # Select the features for the current recycling cycle
            fetch_cur_batch = lambda t: t[..., cycle_no]
            # import pdb;pdb.set_trace()
            # for k,v in batch.items():print(k,v.shape)
            feats = tensor_tree_map(fetch_cur_batch, batch)

            # Enable grad iff we're training and it's the final recycling layer
            is_final_iter = cycle_no == (num_iters - 1)
            with torch.set_grad_enabled(is_grad_enabled and is_final_iter):
                if is_final_iter:
                    # Sidestep AMP bug (PyTorch issue #65766)
                    if torch.is_autocast_enabled():
                        torch.clear_autocast_cache()

                # Run the next iteration of the model
                outputs, m_1_prev, z_prev, x_prev, s_prev = self.iteration(
                    feats,
                    prevs,
                    _recycle=(num_iters > 1)
                )

                if(not is_final_iter):
                    del outputs
                    prevs = [m_1_prev, z_prev, x_prev, s_prev]
                    del m_1_prev, z_prev, x_prev, s_prev

        # if torch.isnan(outputs['single']).any():
        #     print("cryoformer_dec has nan.")
        #     import pdb;pdb.set_trace()
        # Run auxiliary heads
        outputs.update(self.aux_heads(outputs, batch))

        if batch['debug'].sum() and global_step>=0: # training
            if len(batch['batch_idx'].shape)>=2:
                bid = batch['batch_idx'][0,0].item()
            else:
                bid = batch['batch_idx'][0].item()
            if self.training:
                suff = "t"
            else:
                suff = "v"

            name = f"{global_step}_{bid}_r{batch['aatype'].shape[-1]}_{suff}"
            save_dict_as_pdb(batch, outputs, "predictions_in_training2", name)
            # print("save training struture: ", name)
        return outputs
