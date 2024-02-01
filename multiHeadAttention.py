import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def attention(query: torch.Tensor, key: torch.Tensor,
              value: torch.Tensor, mask=None, dropout=None):
    # query, key, value : (B, H, N, C)
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # B, H, N, N
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)
    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return torch.matmul(attn, value), attn  # B, H, N, C

class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, d_model, dropout=0.1):
        super().__init__()
        assert d_model % num_head == 0  # 检查d_model和num_head是否符合要求
        self.num_head = num_head
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])  # q, k, v, linear
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # query, key, value: B, N, C
        # mask: B, N, N
        if mask is not None:
            mask = mask.unsqueeze(1)  # B, 1, N, N
        B, N, C = query.shape
        query, key, value = [l(x).view(B, N, self.num_head, C // self.num_head).transpose(1, 2)  # B, H, N, C
                            for l, x in zip(self.linears, [query, key, value])]
        out, attn = attention(query, key, value, mask, self.dropout)
        out = out.transpose(1, 2).reshape(B, N, C)

        return self.linears[-1](out)        


if __name__ == '__main__':
    x = torch.rand(10, 20, 512)
    num_head = 8
    MSA = MultiHeadAttention(num_head, x.size(-1))
    print(MSA(x, x, x).shape)