import torch
def d_hinge_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor):
    return torch.relu(1 - real_logits).mean() + torch.relu(1 + fake_logits).mean()
def g_hinge_loss(fake_logits: torch.Tensor):
    return -fake_logits.mean()
