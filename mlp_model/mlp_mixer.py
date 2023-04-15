import torch.nn as nn
import einops
from mlp_model.mlp_struct import Mixer_struc

class MLP_Mixer(nn.Module):
    def __init__(self, image_size, patch_size, token_dim, channel_dim, num_classes, dim, num_blocks, dropout=.0):
        super(MLP_Mixer, self).__init__()
        n_patches = (image_size // patch_size) ** 2
        self.patch_size_embbeder = nn.Conv2d(kernel_size=patch_size, stride=patch_size, in_channels=1, out_channels=dim)
        self.blocks = nn.ModuleList([
            Mixer_struc(patches=n_patches, token_dim=token_dim, channel_dim=channel_dim, dim=dim, dropout=dropout) for i in range(num_blocks)
        ])

        self.Layernorm1 = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)
    def forward(self, x):
        out = self.patch_size_embbeder(x)
        out = einops.rearrange(out, "n c h w -> n (h w) c")
        for block in self.blocks:
            out = block(out)
        out = self.Layernorm1(out)
        out = out.mean(dim=1)  # 平均池化
        result = self.classifier(out)
        return result

if __name__ == '__main__':
    model = MLP_Mixer(image_size=45, patch_size=5, dim=512, num_classes=3, num_blocks=8, token_dim=256, channel_dim=2048)
