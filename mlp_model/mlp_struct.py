import torch.nn as nn
from mlp_model.mlp_block import MLPBlock
import einops

class Mixer_struc(nn.Module):
    def __init__(self, patches: int, token_dim: int, dim: int, channel_dim: int, dropout=0.):
        super(Mixer_struc, self).__init__()
        self.patches = patches
        self.channel_dim = channel_dim
        self.token_dim = token_dim
        self.dropout = dropout

        self.MLP_block_token = MLPBlock(patches, token_dim, self.dropout)
        self.MLP_block_chan = MLPBlock(dim, channel_dim, self.dropout)
        self.LayerNorm = nn.LayerNorm(dim)

    def forward(self, x):
        out = self.LayerNorm(x)
        out = einops.rearrange(out, 'b n d -> b d n')
        out = self.MLP_block_token(out)
        out = einops.rearrange(out, 'b d n -> b n d')
        out += x
        out2 = self.LayerNorm(out)
        out2 = self.MLP_block_chan(out2)
        out2 += out
        return out2