import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.beta = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

if __name__ == '__main__':
    x = torch.randn(10, 20, 100)  # B N C
    print(x)
    layer_norm = LayerNorm(x.size(-1))
    print('---------')
    print(layer_norm(x))