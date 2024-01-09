import numpy as np
import matplotlib.pyplot as plt
from ManifoldSculpting import ManifoldSculpting

def gen_roll(n):
    data = np.zeros((n,4))
    t = np.array([2+8*i/n for i in range(n)])
    data[:,0] = t*np.sin(t)
    data[:,2] = t*np.cos(t)
    data[:,1] = np.random.uniform(-6,6,n)
    data[:,3] = t
    return data

data = gen_roll(1000)
roll = data[:,:3].astype(np.float32)
phi = data[:,-1]

ms = ManifoldSculpting(k=12,sigma=0.95,niter=250)
ms.fit(roll)

plt.scatter(ms.pca_data[:,0],ms.pca_data[:,1],c = phi)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1], projection='3d')
ax.scatter3D(ms.pca_data[:,0],ms.pca_data[:,1],ms.pca_data[:,2],c=phi)

plt.show()