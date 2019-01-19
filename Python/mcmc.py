import numpy as np
from tqdm import tqdm

class MCMC(object):

    def __init__(self, target, motion, n_steps):
        self.target = target
        self.motion = motion
        self.n_steps = n_steps

    def _step(self, x):
        y = self.motion.sample(x)
        alpha = np.clip(self.target.ratio_l(y, x), 0, 1)
        mask = (np.random.rand(len(alpha)) < alpha).astype(bool)
        return np.concatenate((y[mask], x[(1-mask).astype(bool)]))

    def forward(self, x):
        for _ in range(self.n_steps):
            x = self._step(x)
        return x
