"""
inverse-distance-weighted interpolation using KDTree:
"""
from __future__ import division
import numpy as np
from scipy.spatial import cKDTree as KDTree


# http://docs.scipy.org/doc/scipy/reference/spatial.html
# ...............................................................................
class InterpolTree:

    def __init__(self, X, Y, leafsize=10, stat=0):
        assert len(X) == len(Y), "len(X) %d != len(z) %d" % (len(X), len(Y))
        self.tree = KDTree(X, leafsize=leafsize)  # build the tree
        self.Y = Y
        self.stat = stat
        self.wn = 0
        self.wsum = None;

    def __call__(self, q, n_near=6, eps=.1, p=3, weights=None):
        # nnear nearest neighbours of each q point --
        q = np.asarray(q)
        qdim = q.ndim
        if qdim == 1:
            q = np.array([q])
        if self.wsum is None:
            self.wsum = np.zeros(n_near)

        self.distances, self.ix = self.tree.query(q, k=n_near, eps=eps)
        interpol = np.zeros((len(self.distances),) + np.shape(self.Y[0]))
        jinterpol = 0
        for dist, ix in zip(self.distances, self.ix):
            if n_near == 1:
                wY = self.Y[ix]
            elif dist[0] < 1e-10:
                wY = self.Y[ix[0]]
            else:  # weight Y s by 1/dist --
                w = 1 / dist ** p
                if weights is not None:
                    w *= weights[ix]  # >= 0
                w /= np.sum(w)
                wY = np.dot(w, self.Y[ix])
                if self.stat:
                    self.wn += 1
                    self.wsum += w
            interpol[jinterpol] = wY
            jinterpol += 1
        return interpol if qdim > 1 else interpol[0]
