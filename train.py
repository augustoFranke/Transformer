import torch
from torch import nn

class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(27,8)
        self.ff = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 8)
        )
        self.attention = nn.MultiheadAttention(embed_dim=8,num_heads=4)
        self.norm1 = nn.LayerNorm(8)
        self.norm2 = nn.LayerNorm(8)
        self.output_proj = nn.Linear(8, 27)
    
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.norm1(x + self.attention(x, x, x)[0])
        x = self.norm2(x + self.ff(x))
        x = x.squeeze(1)
        x = self.output_proj(x)
        return x
    
tensor = torch.randint(0, 27, (4,))

model = TinyTransformer()

print(model(tensor))