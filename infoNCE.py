import torch
import torch.nn as nn
import torch.nn.functional as F

def infoNCE(v_logits: torch.Tensor, w_logits: torch.Tensor):
    # v_logits, w_logits: (B, C)
    v_logits = F.normalize(v_logits, p=2, dim=-1)
    w_logits = F.normalize(w_logits, p=2, dim=-1)
    B, C = v_logits.shape
    sim_logits = torch.matmul(v_logits, w_logits.transpose(-2, -1))  # (B, C)(C, B) -> B,B
    labels = torch.arange(B).long()
    loss_1 = F.cross_entropy(sim_logits, labels)
    loss_2 = F.cross_entropy(sim_logits.transpose(-2, -1), labels)
    return 0.5 * loss_1 + 0.5 * loss_2


if __name__ == '__main__':
    v_logits = torch.randn(10, 64)
    w_logits = torch.randn(10, 64)
    print(infoNCE(v_logits, w_logits))
