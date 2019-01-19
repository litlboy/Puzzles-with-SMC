import numpy as np

class IS(object):

    def __init__(self, target):
        self.target = target
        self.is_fit = False

    def fit(self, particles):
        self.particles = particles
        self.N = self.particles.shape[0]
        weights = self.target.l(particles)
        self.norm_weights = weights/np.sum(weights)
        self.is_fit = True

    def resample(self):
        assert self.is_fit, 'Fit particules first'
        indexes = np.random.choice(np.arange(self.N), p=self.norm_weights, size=len(self.particles))
        return self.particles[indexes, :]
