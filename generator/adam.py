# coding=utf-8
import torch
from torch.optim import Optimizer

class AdamWeightDecayOptimizer(Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay.
    https://github.com/google-research/bert/blob/master/optimization.py
    https://raw.githubusercontent.com/pytorch/pytorch/v1.0.0/torch/optim/adam.py"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        # params = [{'params':weight_decay_params, 'weight_decay':1e-4},
        #                    {'params':no_weight_decay_params, 'weight_decay':0.}]
        # lr-1e-3
        # betas-(0.9, 0.999)
        # eps-1e-6
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        # 使用参数params和defaults初始化Optimizer
        super(AdamWeightDecayOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamWeightDecayOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            # {'amsgrad': False}
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # 提取参数
        # weight_decay_params
        # no_weight_decay_params
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # 求梯度
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                # {'amsgrad': False}
                amsgrad = group['amsgrad']

                # state-p
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                # exp_avg-[]
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                # betas-(0.9, 0.999)
                beta1, beta2 = group['betas']

                # step-迭代次数
                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # beta1*exp_avg+(1-beta1)*grad
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # beta2*exp_avg_sq+(1-beta2)*grad**2
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    # denom-sqrt(beta2*exp_avg_sq+(1-beta2)*grad**2)+eps
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want ot decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # update-exp_avg/denom+weight_decay*p.data
                update = (exp_avg/denom).add_(group['weight_decay'], p.data)
                # 更新参数
                # p.data-lr*update
                p.data.add_(-group['lr'], update)
        return loss