import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from umap import UMAP

def FloydWarshall(data,neigh=5):
    """Floyd Warshall algorithm

    Parameters
    ----------
    data : array or dataframe
        data for the graph
    neigh : int, optional
        number of neighbours, by default 5

    Returns
    -------
    ndarray
        length of the paths
    """
    N = data.shape[0]
    nn = NearestNeighbors(n_neighbors=neigh)
    nn.fit(data)
    temp = nn.kneighbors_graph(mode='distance')
    w = np.full((N,N),np.inf)
    for i,j in zip(temp.nonzero()[0],temp.nonzero()[1]):
        w[i,j] = temp[i,j]
    for i in range(N):
        w[i,i] = 0
    w = np.minimum(w,w.T)
        
    for k in range(N):
        d = (w[:,k])[:,None] + w[k,:]
        w = np.minimum(w,d)
    return w

def isomap(data,d,neigh=5):
    """isomap

    Parameters
    ----------
    data : DataFrame?
        original data, in numpy array (or pandas dataframe?)
    d : int
        target space dimension
    neigh : int, optional
        number of neighbours, by default 5

    Returns
    -------
    NDArray
        transformed data
    NDArray
        eigenvalues
    NDArray
        eigenvectors
    
    """
    delta = FloydWarshall(data,neigh)
    delta = delta * delta
    N = delta.shape[0]
    g = np.zeros((N,N))

    C = np.eye(N) - 1/N * np.ones((N,N))

    g = -0.5 * np.matmul(C,np.matmul(delta,C))

    eigval, eigvec = np.linalg.eig(g)
    idx = np.argsort(-np.abs(eigval))
    lamb = eigval[idx]
    v = eigvec[:,idx]
    lamb_d = np.diag(np.abs(lamb[:d]))
    v_d = v[:,:d]
    y = np.matmul(v_d,np.sqrt(lamb_d))
    
    return y,lamb_d,v_d

def gaussianKernel(xi,xj,sigma):
    return np.exp(-(np.linalg.norm(xi-xj))/(2*sigma*sigma))

def polynomialKernel(xi,xj,delta):
    return (1+np.dot(xi,xj))**delta

def kernelPCA(x,kernel,**kwargs):
    """Kernel PCA

    Parameters
    ----------
    x : DataFrame or NDArray
        data to transform
    kernel : callable
        kernel function

    Returns
    -------
    tuple
        transformed data, eigenvalues, eigenvectors
    """
    if isinstance(x,pd.DataFrame):
        a = x.values
    else:
        a = x
    N = a.shape[0]
    K = np.zeros((N,N))
    # for i in range(N):
    #     for j in range(i,N):
    #         K[i,j] = kernel(a[i,:],a[j,:],**kwargs)
    #         K[j,i] = K[i,j]
    K = pairwise_distances(a,a,kernel,**kwargs)
    
    C = np.eye(N,N) - 1/N * np.ones((N,N))
    K = np.matmul(C,np.matmul(K,C))

    lam,v = np.linalg.eigh(K)
    idx = np.argsort(-np.abs(lam))
    lam = lam[idx]
    v = v[:,idx]
    y = np.matmul(v,np.diag(np.sqrt(np.abs(lam))))

    return y,lam,v


data = np.load('../data/swiss_hole.npy')
roll = data[:,:3].astype(np.float32)
roll = roll - np.mean(roll,axis=0)
phi = data[:,-1]

fig = plt.figure()

## ISOMAP
y,_,_ = isomap(roll,2,neigh=15)
np.save('../data/isomap_embed_hole.npy',y)
ax = fig.add_subplot(221)
ax.scatter(y[:,0],y[:,1],c=phi)
ax.set_title('isomap')

## KERNEL PCA
y,_,_ = kernelPCA(roll,gaussianKernel,sigma=1.1)
np.save('../data/kpca_embed_hole.npy',y)
ax = fig.add_subplot(222)
ax.scatter(y[:,0],y[:,1],c=phi)
ax.set_title('kernel PCA')

## TSNE
tsne = TSNE(perplexity=80)
y = tsne.fit_transform(roll)
np.save('../data/tsne_embed_hole.npy',y)
ax = fig.add_subplot(223)
ax.scatter(y[:,0],y[:,1],c=phi)
ax.set_title('T-SNE')

## UMAP
umap = UMAP(n_neighbors=40,min_dist=0.5)
y = umap.fit_transform(roll)
np.save('../data/umap_embed_hole.npy',y)
ax = fig.add_subplot(224)
ax.scatter(y[:,0],y[:,1],c=phi)
ax.set_title('UMAP')

fig.tight_layout()
plt.show()

