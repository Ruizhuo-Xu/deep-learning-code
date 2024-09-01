import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MHA(nn.Module):
    def __init__(self, d_model, num_head):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        assert self.d_model % self.num_head == 0
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])

    def self_attention(self, q, k, v):
        # B, H, N, C 
        B, H, N, C = q.shape
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(C)
        attn_score = F.softmax(attn_score, dim=-1)
        return torch.matmul(attn_score, v).view(B, N, -1)

    def forward(self, q, k, v):
        B, N, C = q.shape
        q, k, v = [l(x).view(B, N, self.num_head, C // self.num_head).transpose(1, 2)
                   for l, x in zip(self.linears, [q, k, v])]
        out = self.self_attention(q, k, v)
        out = self.linears[-1](out)
        return out

if __name__ == '__main__':
    x = torch.rand(10, 20, 512)
    num_head = 8
    MSA = MHA(x.size(-1), num_head)
    print(MSA(x, x, x).shape)