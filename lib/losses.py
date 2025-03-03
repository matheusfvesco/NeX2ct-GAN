import torch
import torch.nn as nn


class GANLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        self.real_label = 1.0
        self.fake_label = 0.0
        self.loss = nn.MSELoss()
    
    def get_target(self, preds, is_real):
        if is_real:
            label = self.real_label
        else:
            label = self.fake_label
        
        target = torch.zeros(preds.size(), dtype=torch.float).fill_(torch.Tensor(label))
        return target

    def forward(self, preds, is_real):
        target = self.get_target(preds, is_real)
        loss = self.loss(preds, target)
        return loss