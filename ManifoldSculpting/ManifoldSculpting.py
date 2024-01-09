import numpy as np
from sklearn.neighbors import KDTree
import queue
from numba.experimental import jitclass
from numba import int32,float32

specs = [('k',int32),
         ('n_dim',int32),
         ('niter',int32),
         ('sigma',float32),
         ('data',float32[:,:]),
         ('pca_data',float32[:,:]),
         ('neighbours',int32[:,:]),
         ('colinear',int32[:,:]),
         ('delta0',float32[:,:]),
         ('theta0',float32[:,:]),
         ('delta_ave',float32),
         ('d_pres',int32[:]),
         ('d_scal',int32[:])
         ]

@jitclass(specs)
class ManifoldSculpting():
    # general note: might want to pass the data to all functions
    # instead of using self.pca_data
    def __init__(self, k=5, n_dim = 2, niter = 100, sigma = 0.99):

        self.k = k
        self.n_dim = n_dim
        self.niter = niter
        self.sigma = sigma

    def fit(self,data):
        self.data = data
        # find neighbours, distances and angles
        self.neighbours, self.delta0, self.delta_ave, self.colinear, self.theta0 = self._compute_neighbourhood()
        
        # PCA step
        self.pca_data = self._pca()
        self.d_pres = np.array(list(range(self.n_dim)),dtype=np.int32)
        self.d_scal = np.array(list(range(self.n_dim,self.data.shape[1])),dtype=np.int32)

        # adjust variables for given number of iterations
        for iter in range(self.niter):
            # get the origin for the breadth first adjust
            origin = np.random.choice(np.array(list(range(self.data.shape[0])),dtype=np.int32))

            # q = queue.Queue()
            q = []
            q.append(origin)
            visited = []

            # scale the unwanted dimensions
            self.pca_data[:,self.d_scal] *= self.sigma
            # scale the preserved dimensions to preserve average distance.
            # (unclear when this is stated in the paper but authors fo this)
            while self._avg_neighbour_distance() < self.delta_ave:
                self.pca_data[:,self.d_pres] /= self.sigma

            # iterate over items in queue
            while q:
                # get the next point in queue and skip it if already adjusted
                p = q.pop(0)
                if p in visited:
                    continue

                # print(f'\rit:{iter}, p:{p}')
                # enqueue all the point's neighbours
                for n in self.neighbours[p]:
                    q.append(n)

                self._adjust_point(p,visited)
                visited.append(p)

    def compute_error(self,p,visited):
        w = np.zeros(self.k)
        # if the neighbour has already been adjusted, weight is 10. 1 otherwise
        for j in range(self.k):
            w[j] = 10 if self.neighbours[p,j] in visited else 1

        # iterate over all neighbours and compute distances and angles
        total_err = 0
        for i in range(self.k):
            n = self.neighbours[p,i]
            c = int(self.colinear[p,i])
        
            a = self.pca_data[p] - self.pca_data[n]
            b = self.pca_data[c] - self.pca_data[n]
            la = np.linalg.norm(a)
            lb = np.linalg.norm(b)
            theta = np.arccos(np.minimum(1,np.maximum(np.dot(a,b)/(la*lb),-1)))

            err_dist = 0.5*(la-self.delta0[p,i])/self.delta_ave
            err_theta = (theta-self.theta0[p,i])/np.pi
            total_err += w[i] * (err_dist*err_dist + err_theta*err_theta)
        
        return total_err

    def _compute_neighbourhood(self):
        """find neighbours, distances, average distance, colinear points, angle 
        for the data on which MnaifoldSculpting is being trained.

        Returns
        -------
        tuple of numpy.ndarray
            (neighbours, distances, average distance, colinear points, angle)
        """
        N = self.data.shape[0]
        # find the kNN using kdtree
        # tree = KDTree(self.data)
        # _dist, _neigh = tree.query(self.data, self.k+1)
        # _dist = _dist[:,1:]
        # _neigh = _neigh[:,1:]

        x2 = np.sum(self.data*self.data,axis = 1)
        data_t = self.data.T
        xx = self.data@data_t
        dist = np.sqrt(np.abs(x2.reshape((-1,1))-2*xx+x2))

        _neigh = np.zeros((N,self.k),dtype=np.int32)
        _dist = np.zeros((N,self.k),dtype=np.float32)
        for i in range(N):
            _neigh[i] = np.argsort(dist[i])[1:self.k+1]
            _dist[i,:] = dist[i,_neigh[i]]

        # find the most colinear point for each point-neighbour pair
        _colinear = np.zeros((N,self.k),dtype=np.int32)
        _theta = np.zeros((N,self.k),dtype=np.float32)
        # iterate over all points
        for i in range(N):
            # for each point iterate over neighbours
            for nj,j in enumerate(_neigh[i]):
                a = self.data[i] - self.data[j]
                la = np.linalg.norm(a)

                # compute and keep track of all angles between i-j-k, where
                # k are neighbours of j
                angles = np.zeros(self.k)
                for nk,k in enumerate(_neigh[j]):
                    b = self.data[k] - self.data[j]
                    lb = np.linalg.norm(b)
                    angles[nk] = np.arccos(np.minimum(1, np.maximum(np.dot(a,b)/(la*lb), -1)))

                # choose the point such that the angle is the closest to pi
                index = np.argmin(np.abs(angles-np.pi))
                _colinear[i,nj] = _neigh[j,index]
                _theta[i,nj] = angles[index] 
        
        _ave_dist = np.mean(_dist)

        return _neigh, _dist, _ave_dist, _colinear, _theta
    
    
    def _pca(self):
        """perform PCA on data to which ManifoldSculpting is being fit.

        Parameters
        ----------
        data : numpy.ndarray
            array of shape [npoints, nfeatures]

        Returns
        -------
        numpy.ndarray
            data alligned to principal components
        """
        cov = np.cov(self.data.T)
        eigval,eigvec = np.linalg.eig(cov)
        index = np.argsort(-eigval)
        eigvec = eigvec[:,index].astype(np.float32)

        return self.data@eigvec
    
    def _avg_neighbour_distance(self):
        dist = 0
        count = 0
        for p in range(self.pca_data.shape[0]):
            for n in self.neighbours[p]:
                count += 1
                dist += np.linalg.norm(self.pca_data[p]-self.pca_data[n])
        dist /= count
        return dist
    
    def _adjust_point(self,p,visited):
        lr = self.delta_ave * 0.4
        improved = True

        while improved:
            improved = False
            err = self.compute_error(p,visited)

            for d in self.d_pres:
                self.pca_data[p,d] += lr

                newerr = self.compute_error(p,visited)
                if newerr >= err:
                    self.pca_data[p,d] -= 2*lr

                    newerr = self.compute_error(p,visited)
                
                if newerr >= err:
                    self.pca_data[p,d] += lr
                else:
                    improved = True