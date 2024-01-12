import numpy as np
import matplotlib.pyplot as plt
from ManifoldSculpting import ManifoldSculpting
import time

data = np.load('data/swiss_roll.npy')
roll = data[:,:3].astype(np.float32)
roll = roll - np.mean(roll,axis=0)
phi = data[:,-1]

ms = ManifoldSculpting(k=25,sigma=0.98,niter=800,rotate=True,patience=50)

t_start = time.time()
ms.fit(roll)
elapsed = time.time() - t_start
print(f'Did {ms.elapsed_epochs} iterations in {elapsed:.3f} s')
print(f'Final mean error: {ms.last_error:.3e}')
print(f'Best mean error: {ms.best_error:.3e}')

#################
# !!!!!!!!!!!!!
# note to self: DO NOT WRITE ON HOLE3 FOR ANY REASON
# !!!!!!!!!!!!!
# np.save('data/ms_embed_hole4.npy',ms.best_data)

# y = ms.pca_data
fig = plt.figure()
# ax = fig.add_subplot(221)
# ax.scatter(y[:,0],y[:,1],c = phi)
# # ax.set_title('Manifold Sculpting')

# ax.set_title('Last epoch')
# ax.scatter(y[:,0],y[:,1],c = phi)
# ax = fig.add_subplot(222, projection='3d')
# ax.scatter3D(ms.pca_data[:,0],ms.pca_data[:,1],ms.pca_data[:,2],c=phi)
# if ms.d_scal == 0:
#     ax.set_xlim(np.min(roll[:,0]),np.max(roll[:,0]))
# elif ms.d_scal == 1:
#     ax.set_ylim(np.min(roll[:,1]),np.max(roll[:,1]))
# else:
#     ax.set_zlim(np.min(roll[:,2]),np.max(roll[:,2]))



y = ms.best_data
ax = fig.add_subplot(121)
ax.scatter(y[:,0],y[:,1],c = phi)
ax.set_title('Best epoch')
ax = fig.add_subplot(122, projection='3d')
ax.scatter3D(ms.best_data[:,0],ms.best_data[:,1],ms.best_data[:,2],c=phi)
# if ms.d_scal == 0:
#     ax.set_xlim(np.min(roll[:,0]),np.max(roll[:,0]))
# elif ms.d_scal == 1:
#     ax.set_ylim(np.min(roll[:,1]),np.max(roll[:,1]))
# else:
#     ax.set_zlim(np.min(roll[:,2]),np.max(roll[:,2]))

plt.show()