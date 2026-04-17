import torch
import torch.nn as nn
from .spikeformer import SpikeDrivenTransformer


class SpikeDrivenTransformerOCR(SpikeDrivenTransformer):
    def __init__(self, vocab_size, **kwargs):
        super().__init__(num_classes=0, **kwargs)

        self.vocab_size = vocab_size

        self.seq_model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=kwargs["embed_dims"],
                nhead=4,
                dim_feedforward=512,
                batch_first=True
            ),
            num_layers=2
        )

        self.ctc_head = nn.Linear(kwargs["embed_dims"], vocab_size)

    def forward(self, x):
        if len(x.shape) < 5:
             #x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
             x = x.unsqueeze(0).expand(self.T, -1, -1, -1, -1)
        else:
            x = x.transpose(0, 1).contiguous()
        
        block = getattr(self, "block")
        patch_embed = getattr(self, "patch_embed")

        x, _, _ = patch_embed(x)
        
        for blk in block:
            x, _, _ = blk(x)

        # x shape: [T, B, C, H, W]

        # moyenne temporelle
        x = x.mean(0)  # [B, C, H, W]

        # on garde la largeur comme séquence
        x = x.mean(2)  # moyenne sur H → [B, C, W]

        x = x.permute(0, 2, 1)  # [B, W, C]
        
        # Ajout du contexte 
        x = self.seq_model(x)

        logits = self.ctc_head(x)  # [B, W, vocab]

        return logits
