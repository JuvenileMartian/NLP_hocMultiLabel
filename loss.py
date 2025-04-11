import torch
from torch import nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma 
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, labels):
        bce_loss = self.bce_loss(logits, labels)
        probs = torch.sigmoid(logits)

        probs = probs * labels + (1-probs) * (1-labels)

        focal_loss = self.alpha * (1 - probs) ** self.gamma * bce_loss
        return focal_loss.mean()