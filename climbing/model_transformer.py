import torch
from torch import nn
import random

from einops import rearrange
from einops.layers.torch import Rearrange

from vit_pytorch.simple_vit import pair, Transformer, posemb_sincos_2d

def kilter_posemb_sincos_2d(patches, patches_pos, temperature = 10000, dtype = torch.float32):
    # _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype
    dim, device, dtype = patches.shape[-1], patches.device, patches.dtype
    x, y = patches_pos[..., (0,)], patches_pos[..., (1,)]

    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y * omega[None, None, :]
    x = x * omega[None, None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=-1)
    return pe.type(dtype)

class LastLayer(nn.Module):
    def forward(self, x):
        # regression
        x = torch.sigmoid(x)
        # allowing it to reach 39 easily
        x = x*50
        return x

class ClimbingSimpleViT(nn.Module):
    # adapted from from vit_pytorch.simple_vit.SimpleVit

    def __init__(self, *, hold_data_len, dim, depth, heads, mlp_dim, dim_head = 64):
        """
        hold_data_len includes the two positional values
        """
        super().__init__()

        self.to_angle_embedding = nn.Sequential(
            nn.Linear(1, dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(dim//2, dim),

        )
        self.to_holds_embedding = nn.Sequential(
            nn.Linear(hold_data_len-2, dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(dim//2, dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
            LastLayer(),
        )

    def forward(self, data_in):
        angle = data_in[..., (0,), 0][..., None] # only first value of angle is useful.
        holds_pos = data_in[..., 1:, 0:2]
        holds = data_in[..., 1:, 2:]

        # normalize angle (not sure this is the best way but should work)
        angle -= 30
        angle /= 180.0

        # TODO: find better way to insert angle data
        # TODO: try 3D positional embeddings instead of one angle value?
        # TODO: try adding angle embedding to holds embedding?
        angle = self.to_angle_embedding(angle)
        holds = self.to_holds_embedding(holds)

        # Add positional embedding to holds
        positional_encoding = kilter_posemb_sincos_2d(holds, holds_pos)
        holds = holds + positional_encoding

        x = torch.concat([angle, holds], dim=-2)
        # data_in holds has been padded with some -1 values to mark end of sentence.
        # we find this index and replace values with zeroes.
        for i in range(data_in.shape[0]):
            sep_pos = (data_in[i, ..., 0] == -1).nonzero(as_tuple=True)[0][0]
            # TODO: validate that doing this makes sense for a ViT (check what BERT does with the SEP and CLS tokens)
            # SEP
            x[i, sep_pos, 0] = 1  
            x[i, sep_pos, 1:] = 0
            # PAD
            x[i, sep_pos+1:, :] = 0  

        x = self.transformer(x)
        # TODO: check if using mean here is usual. It reduces from [b, n_holds, 32] to [b, 32]
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        x = self.linear_head(x)
        return x

if __name__ == "__main__":
    ### example
    v = ClimbingSimpleViT(
        hold_data_len=2+1,
        dim = 512,
        depth = 6,
        heads = 16,
        mlp_dim = 1024
    )
    batch_size = 5
    hold_data_len = 2+1  # x, y, HOLD_TYPE
    angle = torch.randn(batch_size, 1, hold_data_len) * 100
    angle[..., 1:] = -1  # not required but helps to debug
    holds = torch.randn(batch_size, 31, hold_data_len)
    for i in range(batch_size):
        # randomply place sentense stop token
        sep_pos = random.randint(5, 30)
        holds[i, sep_pos:, :] = -1

    x = torch.concat([angle, holds], dim=-2)

    preds = v(x)