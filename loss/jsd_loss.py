import torch
import torch.nn.functional as F

EPISILON=1e-10

class JSDLoss(torch.nn.Module):

  def __init__(self, weight=1.0, softmax_sign=False):
    super(JSDLoss, self).__init__()
    self.weight = weight
    self.softmax_sign = softmax_sign

  def forward(self, p, q):
    if self.softmax_sign is False:
      p = F.softmax(p, dim=1)
      q = F.softmax(q, dim=1)

    loss1 = p * torch.log(p / q + EPISILON)
    loss1 = loss1.sum(1)

    loss2 = q * torch.log(q / p + EPISILON)
    loss2 = loss2.sum(1)

    loss = loss1 + loss2
    loss = loss.mean()*self.weight
    return loss
