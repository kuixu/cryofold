
from typing import Optional

import torch
import torch.nn as nn

from cryofold.model.primitives import Linear, LayerNorm
from cryofold.utils.chunk_utils import chunk_layer


class PairTransition(nn.Module):
    """
    Implements Algorithm 15.
    """

    def __init__(self, c_z, n):
        """
        Args:
            c_z:
                Pair transition channel dimension
            n:
                Factor by which c_z is multiplied to obtain hidden channel
                dimension
        """
        super(PairTransition, self).__init__()

        self.c_z = c_z
        self.n = n

        self.layer_norm = LayerNorm(self.c_z)
        self.linear_1 = Linear(self.c_z, self.n * self.c_z, init="relu")
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.n * self.c_z, c_z, init="final")

    def _transition(self, z, mask):
        # [*, N_res, N_res, C_z]
        z = self.layer_norm(z)
        
        # [*, N_res, N_res, C_hidden]
        z = self.linear_1(z)
        z = self.relu(z)

        # [*, N_res, N_res, C_z]
        z = self.linear_2(z) 
        z = z * mask

        return z

    @torch.jit.ignore
    def _chunk(self,
        z: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        return chunk_layer(
            self._transition,
            {"z": z, "mask": mask},
            chunk_size=chunk_size,
            no_batch_dims=len(z.shape[:-2]),
        )

    def forward(self, 
        z: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        # DISCREPANCY: DeepMind forgets to apply the mask in this module.
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        # [*, N_res, N_res, 1]
        mask = mask.unsqueeze(-1)

        if chunk_size is not None:
            z = self._chunk(z, mask, chunk_size)
        else:
            z = self._transition(z=z, mask=mask)

        return z
