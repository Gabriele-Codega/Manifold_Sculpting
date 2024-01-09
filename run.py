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

data = gen_roll(2000)
roll = data[:,:3].astype(np.float32)
roll = roll - np.mean(roll,axis=0)
phi = data[:,-1]

ms = ManifoldSculpting(k=25,sigma=0.99,niter=500,rotate=True)
ms.fit(roll)
print('inital lr = ',ms.delta_ave)
print('final lr = ',ms.learning_rate)

plt.scatter(ms.pca_data[:,0],ms.pca_data[:,1],c = phi)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1], projection='3d')
ax.scatter3D(ms.pca_data[:,0],ms.pca_data[:,1],ms.pca_data[:,2],c=phi)
if ms.d_scal == 0:
    ax.set_xlim(np.min(roll[:,0]),np.max(roll[:,0]))
elif ms.d_scal == 1:
    ax.set_ylim(np.min(roll[:,1]),np.max(roll[:,1]))
else:
    ax.set_zlim(np.min(roll[:,2]),np.max(roll[:,2]))

# p = 500
# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1],projection='3d')
# ax.scatter3D(ms.data[p-40:p+40,0],ms.data[p-40:p+40,1],ms.data[p-40:p+40,2],c='k',alpha=0.3)
# ax.scatter3D(ms.data[p,0],ms.data[p,1],ms.data[p,2],c='r',marker='*')
# ax.scatter3D(ms.data[ms.neighbours[p],0],ms.data[ms.neighbours[p],1],ms.data[ms.neighbours[p],2],c=ms.neighbours[p],alpha=1,marker='o',cmap='turbo')
# ax.scatter3D(ms.data[ms.colinear[p],0],ms.data[ms.colinear[p],1],ms.data[ms.colinear[p],2],c=ms.neighbours[p],marker='^',alpha=1)
# ax.set_aspect('equal')

plt.show()