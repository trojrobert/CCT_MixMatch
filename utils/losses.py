import torch.nn.functional as F
from utils import ramps 



class consistency_weight:
    """ Manage iteration and epoch process """

    def __init__(self, final_w, iters_per_epoch, rampup_starts=0,
                     rampup_ends=7, ramp_type='sigmoid_rampup'):
        self.final_w = final_w 
        self.iters_per_epoch = iters_per_epoch
        self.rampup_starts = rampup_starts * iters_per_epoch
        self.rampup_ends = rampup_ends *  iters_per_epoch
        self.rampup_length = (self.rampup_ends - self.rampup_starts)
        self.rampup_func = getattr(ramps, ramp_type)
        self.current_rampup = 0 


    def __call__(self, epoch, curr_iter): 

        cur_total_iter = self.iters_per_epoch * epoch + curr_iter
        if curr_total_iter < self.rampup_starts: 
            return 0 
        self.current_rampup =  self.rampup_func(cur_total_iter - self.rampup_starts, self.rampup_length)
        return self.final_w * self.current_rampup


def CE_loss(input_logits, target_targets, ignore_index, temperatute=1):
    return F.cross_entropy(input_logits/temperatute, target_targets, ignore_index=ignore_index)

def softmax_mse_loss(inputs, target, conf_mask=False, threshold=None, use_softmax=False):
    """Calculate MSE Loss

    Args:
        inputs() :
        target() :
        conf_mask():
        threshold():
        use_softmax():

    Return:

    """

    assert inputs.required_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size()

    inputs = F.sofmax(inputs, dim=1)

    if use_softmax:
        target = F.softmax(targets, dim=1)

    if conf_mask:
        loss_mat = F.mse_loss(inputs, targets, reduction='none')
        mask = (targets.max(1)[0] > threshold)
        loss_mat = loss_mat[mask.unsqueeze(1).expand_as(loss_mat)]

        if loss_mat.shape.numel() == 0: loss_mat = torch.tensor([0.]).to(inputs.device)
        return loss_mat.mean()

    else:
        return F.mse_loss(inputs, targets, reduction='mean')