from puzzle import Puzzle
from importance_sampling import IS
from distribs import Distribution
from mcmc import MCMC

import copy
import numpy as np
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed
import pickle

class TargetLoss(Distribution):

    def __init__(self, loss, lambd):
        Distribution.__init__(self)
        self.loss = loss
        self.lambd = lambd

    # Avoid 0/0 division for Metropolis step
    def ratio_l(self, x, y):
        return np.exp(-self.lambd*(self.loss(x)-self.loss(y)))

    def l(self, x):
        return np.exp(-self.lambd*(self.loss(x))) + 1e-300


class Kernel(Distribution):
    def __init__(self, t):
        Distribution.__init__(self)
        self.t = t

    def _flip(self, x):
        flip = np.random.rand()

        if flip > 0.75*((0.9)**self.t):
            i, j = np.random.choice(len(x), size=2, replace=False)
            y = copy.copy(x)
            y[i], y[j] = x[j], y[i]

        else:
            i = np.random.randint(len(x))
            y = []
            y += x[i:].tolist()
            y += x[:i].tolist()
            return np.array(y)

        return y

    def sample(self, x):
        transpositions = np.array([self._flip(permutation) for permutation in x])
        return transpositions


########################################################################################################################

# Parameters of puzzle
n = 10
k = 2

# SMC parameters
N = 100
n_steps = 50
max_iter = 10

def SMC(n, k, N, n_steps, verbose=False):

    # Trigger timer
    start = time.time()

    # Build the puzzle
    puzzle = Puzzle(n, k)
    print('Number of pieces in puzzle %s' % puzzle.d)

    # Initialize the algorithm
    particles = (np.ones((N, puzzle.d))*np.arange(puzzle.d)).astype(int)
    particles = np.array([np.random.permutation(particles[i][:]) for i in range(N)])
    t = 0
    av_loss = []

    # Annealing kernel
    kernel = Kernel(t)

    # Compute initial loss
    av_loss.append(np.mean(puzzle.loss_global(particles)))
    if verbose:
        print('Initial average number of non matching edges: %s' % av_loss[-1])
        print('')

    while t < max_iter:
        if verbose:
            print('Step %s' % (t+1))
        # Annealing loss
        loss = TargetLoss(puzzle.loss, 2*np.log(t + np.exp(1)))

        # Re-sample for importance weights
        isampler = IS(loss)
        isampler.fit(particles)
        particles = isampler.resample()

        # Forward search
        mcmc = MCMC(loss, kernel, n_steps)
        particles = mcmc.forward(particles)

        # Track performance
        av_loss.append(np.mean(puzzle.loss_global(particles)))
        if verbose:
            print('Current average number of non matching edges: %s' % av_loss[-1])
            print('------------------------------------------------------------------')

        # Update t
        t += 1

    end = time.time()

    return av_loss, t, end-start


if __name__ == '__main__':

    #res = Parallel(n_jobs=-1)(delayed(SMC)(n, k, N, n_steps) for _ in range(10))

    #pickle.dump(res, open('Sim_results', 'wb'))

    #SMC(n, k, N, n_steps, True)

    res = Parallel(n_jobs=-1)(delayed(SMC)(n, k, N, n_steps, True) for _ in range(10))

    pickle.dump(res, open('loc_global_results', 'wb'))

    #plt.plot(np.arange(len(means)), means)
    #plt.fill_between(np.arange(len(means)), means - (1.96/3.5)*vars, means + (1.96/3.5)*vars)
    #plt.xlabel('Iteration')
    #plt.ylabel('Number of matching edges')
    #plt.show()
