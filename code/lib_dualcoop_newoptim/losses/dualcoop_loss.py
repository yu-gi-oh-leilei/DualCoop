import torch
import torch.nn as nn

class AsymmetricLoss_partial(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, 
                 disable_torch_grad_focal_loss=True,
                 thresh_pos=0.9, thresh_neg=-0.9):
        super(AsymmetricLoss_partial, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        
        self.thresh_pos = thresh_pos
        self.thresh_neg = thresh_neg

        self.margin = 0.0

    def forward(self, x, y, if_partial=True):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # positive_mask = (y > self.margin).float()
        # negative_mask = (y < -self.margin).float()

        y_pos = (y > self.thresh_pos).float()
        y_neg = (y < self.thresh_neg).float()
        # Basic CE calculation
        los_pos = y_pos * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = y_neg * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y_pos
            pt1 = xs_neg * y_neg  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y_pos + self.gamma_neg * y_neg
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum() / x.shape[0] if if_partial else -loss.mean()


# def dualcoop_loss(inputs, inputs_g, targets):
#     """
#     using official ASL loss.
#     """
#     loss_fun = AsymmetricLoss_partial(gamma_neg=2, gamma_pos=1, clip=0.05)

#     return loss_fun(inputs, targets) # + loss_fun(inputs_g, targets)