import torch
import torch.nn.functional as F
import torch.nn as nn

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
):
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class TStage_FocalLoss(nn.Module):
    '''
    This loss is intended for single-label classification problems
    '''
    def __init__(self, gamma_pos=0, gamma_neg=4, gamma_lc=2, gamma_hc=1, eps: float = 0.1, reduction='mean', epochs=50,
                 factor=1.0, alpha=0.5):
        super(TStage_FocalLoss, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_hc = gamma_hc
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.gamma_lc = gamma_lc
        self.reduction = reduction
        self.epochs = epochs
        self.alpha = alpha
        self.factor = factor # factor=2 for cyclical, 1 for modified
#       self.ceps = ceps
#       print("Asymetric_Cyclical_FocalLoss: gamma_pos=", gamma_pos," gamma_neg=",gamma_neg,
#             " eps=",eps, " epochs=", epochs, " factor=",factor)

    def forward(self, inputs, target, epoch, reduction=None):
        '''
        "input" dimensions: - (batch_size, number_classes)
        "target" dimensions: - (batch_size)
        '''
#        print("input.size(),target.size()) ",inputs.size(),target.size())
        # Cyclical Focal Loss
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, target, reduction="none"
        )
        p = torch.sigmoid(inputs)
        p_t = p * target + (1-p) * (1-target)
        alpha = self.alpha
        if self.factor*epoch < self.epochs :
            eta = 1.0 - self.factor * epoch *  1.0/ (1.0 * self.epochs)
        else:
            eta = (self.factor*epoch/(1.0 * self.epochs) - 1.0)/(self.factor - 1.0)
        # 最开始两个参数： 30.0 5.0
        loss = ( eta * ( pow(4*torch.maximum(p_t-0.5,torch.tensor(0)),7) + pow(0.5 * (1-p_t),2)) / 30.0  + (1-eta) * ((1 - p_t) ** self.gamma_lc) / 5.0)* ce_loss
        # y = -1 * (-2 * pow(p_t-0.5,2) + 2 * 0.5 * 0.5) * np.log(p_t)
        # loss = ( eta * ( pow(4*torch.maximum(p_t-0.5,torch.tensor(0)),5) + pow(0.5 * (1-p_t),2)) / 20.0  + (1-eta) * ((-2 * pow(p_t-0.5,2) + 2 * 0.5 * 0.5) )) * ce_loss
        # y = -1 * (-0.02 * ((x-0.8)**2) + 0.04*5*5) * np.log(x)
        # 注意下面的损失函数中loss中gamma_hc=2
        if alpha >= 0:
            alpha_t = alpha * target + (1 - alpha) * (1 - target)
            loss = alpha_t * loss

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss

class ASL_FocalLoss(nn.Module):
    '''
    This loss is intended for single-label classification problems
    '''
    def __init__(self, gamma=2, eps: float = 0.1, reduction='mean'):
        super(ASL_FocalLoss, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma = gamma
        self.reduction = reduction
        print("ASL_FocalLoss: gamma=", gamma, " eps=",eps)

    def forward(self, inputs, target):
        '''
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size)
        '''
        # print('inputs.shape: {}'.format(inputs.shape))
        self.targets_classes = target
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        # 注意到这里 targets + anti_targets = 1
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                 self.gamma * targets + self.gamma * anti_targets)
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(self.eps / num_classes)

        # loss calculation
        loss = - 5000 * self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss


class Cyclical_FocalLoss(nn.Module):
    '''
    This loss is intended for single-label classification problems
    '''
    def __init__(self, gamma_pos=0, gamma_neg=4, gamma_hc=0, eps: float = 0.1, reduction='mean', epochs=20,
                 factor=2, alpha=10):
        super(Cyclical_FocalLoss, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_hc = gamma_hc
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction
        self.epochs = epochs
        self.alpha = alpha
        self.factor = factor # factor=2 for cyclical, 1 for modified
#       self.ceps = ceps
        print("Asymetric_Cyclical_FocalLoss: gamma_pos=", gamma_pos," gamma_neg=",gamma_neg,
              " eps=",eps, " epochs=", epochs, " factor=",factor)

    def forward(self, inputs, target, epoch):
        '''
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size)
        '''
#        print("input.size(),target.size()) ",inputs.size(),target.size())
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = target

        # Cyclical
#        eta = abs(1 - self.factor*epoch/(self.epochs-1))
        if self.factor*epoch < self.epochs:
            eta = 1 - self.factor * epoch/(self.epochs-1)
        else:
            eta = (self.factor*epoch/(self.epochs-1) - 1.0)/(self.factor - 1.0)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)
        positive_w = torch.pow(1 + xs_pos + xs_neg, self.gamma_hc * targets)
        log_preds = log_preds * ((1 - eta)* asymmetric_w + eta * positive_w)

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(self.eps / num_classes)

        # loss calculation
        # 这里的self_target_classes代表的是真实类别，这里相乘的目的就在于
        loss = - 100 * self.targets_classes.mul(log_preds)
        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss
