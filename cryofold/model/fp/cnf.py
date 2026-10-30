import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal
import math
from cryofold.utils.fp.structure_utils import get_edge_features
from cryofold.utils.fp.sidechain_utils import Idealizer
from cryofold.utils.fp.so2_utils import exp
from torch_cluster import radius_graph, knn_graph

class CNF(nn.Module):
    def __init__(self, model, config, eps=1e-3, coeff=8.0, stepsize=100, method='euler', mode='vf'):
        super().__init__()
        self.model = model
        self.config = config
        self.eps = eps
        self.coeff = coeff
        self.stepsize = stepsize
        self.method = method
        self.mode = mode

        self.idealizer = Idealizer(use_native_bb_coords=True)

    def forward(self, *args, **kwargs) -> Tensor:
        return self.model(*args, **kwargs)

    def get_vf(self, t, xt, batch):
        dihedral_sin_cos = torch.cat([batch.bb_dihedral.sin(), batch.bb_dihedral.cos()], dim=-1)
        node_feat = torch.cat([batch.aa_onehot, dihedral_sin_cos], dim=-1)

        x_scaled = (xt - math.pi) * batch.chi_mask
        noised_crds = self.idealizer(batch.aa_num, batch.pos[:, :4], x_scaled)  # rescale back to [-pi,pi]
        noised_crds = noised_crds * batch.atom_mask.unsqueeze(-1)

        # edge index
        ca = batch.pos[:, 1]
        if self.config.model.use_virtual_cb:
            b = batch.pos[:, 1, :] - batch.pos[:, 0, :]
            c = batch.pos[:, 2, :] - batch.pos[:, 1, :]
            a = torch.cross(b, c, dim=-1)
            node_crds = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + batch.pos[:, 1, :]
        else:
            node_crds = ca

        if self.config.data.edge_type == 'radius':
            edge_index = radius_graph(node_crds, r=self.config.data.max_radius,
                                      max_num_neighbors=self.config.data.max_neighbors, batch=batch.batch)
        elif self.config.data.edge_type == 'knn':
            edge_index = knn_graph(node_crds, k=self.config.data.max_neighbors, batch=batch.batch)
        else:
            raise NotImplementedError('wrong edge type')

        edge_feat = get_edge_features(noised_crds, edge_index, batch.atom_mask,
                                      all_atoms=self.config.model.add_dist_to_edge)
        pred_chi = self(t, xt, node_feat, node_crds, edge_index, edge_feat, batch.chi_mask, batch.batch, atom_mask=batch.atom_mask,
                        self_cond=None)

        return pred_chi


    def decode(self, batch, return_traj=False, refine_model=None) -> Tensor:
        ca_crds, chi, chi_alt, batch_id, chi_mask, atom_mask = batch.pos[:,1], batch.chi, batch.chi_alt, batch.batch, batch.chi_mask, batch.atom_mask
        chi = (chi + math.pi) * chi_mask
        x0 = torch.rand_like(chi) * 2*math.pi # / 10 # uniform prior [0,2pi]
        x0[batch.chi_alt_mask==1] = x0[batch.chi_alt_mask==1] % math.pi

        x0 = x0 * chi_mask
        t_range = torch.linspace(0, 1.0-self.eps, self.stepsize).to(chi)

        traj = [x0]
        t1 = t_range[0]
        # prev_chi = None
        dropout = torch.rand_like(chi[:,0]) > 0.75

        for t2 in t_range[1:]:
            dt = t2 - t1
            t = torch.ones(x0.shape[0], 1, device=x0.device) * t2
            t1 = t2
            for r in range(1): # multiround sampling
                xt = traj[-1]
                # HARDCODED CONDITIONING
                # xt[dropout] = chi[dropout]
                # cdrh3_mask = batch.cdr_mask == 3
                # xt[~cdrh3_mask] = xt[~cdrh3_mask]
                # ab_mask = batch.chain_mask != 2
                # xt[~ab_mask] = xt[~ab_mask]

                pred_chi_vf = self.get_vf(t, xt, batch)

                if self.mode == 'vf':
                    pred_chi_vf = self.coeff * pred_chi_vf * (1 - t) # exp schedule
                    xt_next = self.euler_step(dt, xt, pred_chi_vf, batch.chi_alt_mask)

                    # xt_next[dropout] = chi[dropout]
                    # xt_next = self.rk4_step(xt, pred_chi_vf, t, dt, batch)

                    # if t[0].item() < 0.5:
                    #     noise = (torch.rand_like(xt_next) * (2*math.pi) * (1-t)) * chi_mask
                    #     xt_next = self.euler_step(dt, xt, noise, batch.chi_alt_mask)

                    # # heun's method
                    # y = self.euler_step(dt, xt, pred_chi_vf, batch.chi_alt_mask)
                    # y_ = self(t, y, node_feat, node_crds, edge_index, edge_feat, chi_mask, batch.batch,
                    #                 atom_mask=atom_mask, self_cond=prev_chi)
                    #
                    # vf_heun = pred_chi_vf + y_
                    #
                    # xt_next = self.euler_step(dt/2, xt, vf_heun, batch.chi_alt_mask)

                elif self.mode == 'cfm':
                    xt_next = xt + (dt * pred_chi_vf)
                    xt_next = torch.remainder(xt_next, 2*math.pi)
                    # xt_next[batch.chi_alt_mask==1] = torch.remainder(xt_next[batch.chi_alt_mask==1], math.pi)
                else: raise NotImplementedError('wrong mode')
                traj.append(xt_next)

            # prev_chi = pred_chi

        if return_traj:
            return traj
        else:
            return traj[-1], dropout # just return last step

    def euler_step(self, dt, xt, vf, alt_mask):
        return exp(xt, vf * dt, alt_mask=alt_mask)

    def rk4_step(self, xt, vt, t0, dt, batch):
        k1 = vt
        k2 = self.get_vf(t0 + dt / 3, xt + dt * k1 / 3, batch)
        k3 = self.get_vf(t0 + dt * 2 / 3, xt + dt * (k2 - k1 / 3), batch)
        k4 = self.get_vf(t0 + dt, xt + dt * (k1 - k2 + k3), batch)
        k5 = xt + (k1 + 3 * (k2 + k3) + k4) * dt * 0.125
        k5 = k5 % (2*math.pi)
        k5[batch.chi_alt_mask == 1] = k5[batch.chi_alt_mask == 1] % math.pi
        return k5

    # Hutchinson trace estimation
    def log_prob(self, x: Tensor) -> Tensor:
        eps = torch.randn_like(x)
        def augmented(t: Tensor, x: Tensor, ladj: Tensor) -> Tensor:
            with torch.enable_grad():
                x = x.requires_grad_()
                dx = self(t, x)

            jacobian = torch.autograd.grad(dx, x, eps)[0]
            trace = torch.sum(jacobian * eps, dim=tuple(range(1, len(x.shape))))

            return dx, trace

        ladj = torch.zeros_like(x[...,0]) # just placeholder for likelihood
        z, ladj = odeint(augmented, (x,ladj), 0.0, 1.0, phi=self.parameters())

        return Normal(0.0, z.new_tensor(1.0)).log_prob(z).sum(dim=-1) + ladj
