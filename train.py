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
        self.pos_embedding = nn.Embedding(10, 8)
    
    def forward(self, x):
        x = self.embedding(x)
        positions = torch.arange(x.shape[0])
        x = (x + self.pos_embedding(positions)).unsqueeze(1)
        x = self.norm1(x + self.attention(x, x, x)[0])
        x = self.norm2(x + self.ff(x))
        x = x.squeeze(1)
        x = self.output_proj(x)
        return x

text = "hello world"
chars = sorted(set(text))
vocab_size = len(chars)
char_to_idx = {c: i for i, c in enumerate(chars)}
token_dict = torch.tensor([char_to_idx[c] for c in text])

x_train = token_dict[:-1]
y_train = token_dict[1:]

model = TinyTransformer()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(500):
    logits = model(x_train)
    loss = criterion(logits, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"epoch {epoch}, loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    logits = model(x_train)
    predictions = torch.argmax(logits, dim=1)
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    predicted_text = ''.join([idx_to_char[i.item()] for i in predictions])
    print("input:    ", text[:-1])
    print("predicted:", predicted_text)