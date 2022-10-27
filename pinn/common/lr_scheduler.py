# !usr/bin/env python
# -*- coding:utf-8 _*-

from collections import Counter
import numpy as np


class _LRScheduler:
    """
    Basic class for learning rate scheduler
    """

    def __init__(self, lr, max_epoch, steps_per_epoch):
        self.base_lr = lr
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = int(max_epoch * steps_per_epoch)

    def get_lr(self):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError


class MultiStepLR(_LRScheduler):
    """
    Multi-step learning rate scheduler

    Decays the learning rate by gamma once the number of epoch reaches one of the milestones.

    Args:
        lr (float): Initial learning rate which is the lower boundary in the cycle.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
        steps_per_epoch (int): The number of steps per epoch to train for.
        max_epoch (int): The number of epochs to train for.

    Outputs:
        numpy.ndarray, shape=(1, steps_per_epoch*max_epoch)

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = MultiStepLR(lr=0.1, milestones=[30,80], gamma=0.1, steps_per_epoch=5000, max_epoch=90)
        >>> lr = scheduler.get_lr()
    """

    def __init__(self, lr, milestones, gamma, steps_per_epoch, max_epoch):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        super(MultiStepLR, self).__init__(lr, max_epoch, steps_per_epoch)

    def get_lr(self):
        lr_each_step = []
        current_lr = self.base_lr
        for i in range(self.total_steps):
            cur_ep = i // self.steps_per_epoch
            if i % self.steps_per_epoch == 0 and cur_ep in self.milestones:
                current_lr = current_lr * self.gamma
            lr = current_lr
            lr_each_step.append(lr)
        return np.array(lr_each_step).astype(np.float32)
