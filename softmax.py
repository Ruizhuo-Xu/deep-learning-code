import torch
import torch.nn as nn
import torch.nn.functional as F
def softMax(logits: torch.Tensor):
    # x: (B N)
    max_logit = torch.max(logits, dim=-1, keepdim=True).values  # 减去最大值防止上溢
    exp_logits = torch.exp(logits - max_logit)  # 计算指数
    return exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)  # 指数 / 总和

def crossEntropy(logits, labels):
    # logits: (B, N), labels: (B,)
    batch_size, num_classes = logits.shape
    p_logits = softMax(logits)
    one_hot_labels = F.one_hot(labels, num_classes=num_classes).float()  # B N
    log_p = torch.log(p_logits)
    loss = -torch.sum(one_hot_labels * log_p) / batch_size
    return loss
    


if __name__ == '__main__':
    x = torch.tensor([2.0, 1.0, 0.1])
    print(f'x={x}')
    print('------------------')
    print(softMax(x))
    print('------------------')
    print(F.softmax(x, dim=-1))
    print('------------------')
    pred=torch.tensor([[0.2,0.3,0.5],[0.3,0.2,0.5],[0.4,0.4,0.2]])
    target=torch.tensor([1,0,2])
    print(crossEntropy(pred, target))
    print('-----------------')
    print(F.cross_entropy(pred, target))


