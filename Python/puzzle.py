import numpy as np
import matplotlib.image as mpimg

class Puzzle(object):

    def __init__(self, n, k):
        assert n % k == 0, 'k must divide n'
        self.d = (n//k)**2
        self.w = n//k
        self.label = np.random.permutation(np.arange(self.d))
        self.signature = self._signature(self.label)
        #self.corners_idx = [0, self.w-1, self.d - self.w, self.d-1]
        #self.borders_idx = np.arange(1, self.w-1).tolist() + (self.w*np.arange(1, self.w-1)).tolist() \
                           #+ (self.w *np.arange(2, self.w)-1).tolist() \
                           #+ np.arange(self.corners_idx[2]+1, self.corners_idx[3]).tolist()
        #self.inners_idx = list(set(set(np.arange(self.d)) - set(self.corners_idx)) - set(self.borders_idx))
        #self.edges = self._perm2edges(self.label.reshape((1, -1)))


    #def _perm2edges(self, permutations):
        #edges = np.repeat(permutations, 3).reshape((-1, permutations.shape[1], 3))
        #edges[:, self.inners_idx, 2:] = 0
        #return edges

    def _signature(self, permutation):
        grid = permutation.reshape((self.w, self.w))/self.d
        exp_grid = np.exp(grid)
        row_signature = np.diff(exp_grid, n=1, axis=0).flatten()
        col_signature = np.diff(exp_grid, n=1, axis=1).flatten()
        return set(row_signature), set(col_signature)

    def _loss(self, permutation):
        signature = self._signature(permutation)
        return len(signature[0] - self.signature[0]) + len(signature[1] - self.signature[1])

    def loss_global(self, permutations):
        loss = np.zeros(permutations.shape[0])
        #shift = self._perm2edges(permutations) - self.edges
        shift = permutations - self.label
        values, nbrs = np.unique(np.where(shift != 0)[0], return_counts=True)
        loss[values] = nbrs
        return loss.astype(float)

    def loss_local(self, permutations):
        return np.array([self._loss(permutation) for permutation in permutations])

    def loss(self, permutations):
        return self.loss_global(permutations) + self.loss_local(permutations)
        #return self.loss_global(permutations)

if __name__ == '__main__':
    puzzle = Puzzle(10,2)
    print(puzzle.loss([np.arange(25)]))
