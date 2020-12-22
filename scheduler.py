from __future__ import division
from __future__ import print_function

# This file defines Scheduler,
# which is used to adjust learning rate during training dynamically.
# Common used strategies are:
#   - decay learning rate by a constant every certain epochs
#   - decay learning rate by a constant when reach some milestones
#   - decay learning rate when some metric is not getting better (reduce on plateau)
# Here I implement the first strategy as StepScheduler.

class Scheduler(object):
    def __init__(self, init_lr):
        self.init_lr = init_lr

    def step(self, epoch):
        pass


class EmptyScheduler(Scheduler):
    def __init__(self, init_lr):
        super(EmptyScheduler, self).__init__(init_lr)

    def step(self, epoch):
        # does nothing
        return self.init_lr


class StepScheduler(Scheduler):
    def __init__(self, init_lr, step_size=10, decay=0.1, min_lr=1e-8):
        super(StepScheduler, self).__init__(init_lr)
        self.step_size = step_size
        self.decay = decay
        self.min_lr = min_lr

    def step(self, epoch):
        # reduce learning rate by [decay] every [step_size] epochs until reaching [min_lr]
        return max(self.init_lr * self.decay**(epoch // self.step_size), self.min_lr)