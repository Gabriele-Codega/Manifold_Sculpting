import numpy as np
import matplotlib.pyplot as plt

def gen_roll(n):
    """Generate swiss roll data

    Parameters
    ----------
    n : int
        number of points

    Returns
    -------
    ndarray
        first three columns are x,y,z coordinates of points.
        Fourth column is the angle of points in cylindrical coordinates.
    """
    data = np.zeros((n,4))
    t = np.array([2+8*i/n for i in range(n)])
    data[:,0] = t*np.sin(t)
    data[:,2] = t*np.cos(t)
    data[:,1] = np.random.uniform(-6,6,n)
    data[:,3] = t
    return data

data = gen_roll(500)

r = np.sqrt(data[:,0]**2+data[:,1]**2)
mask = np.logical_and(r < 3, data[:,2] > 3)

data_hole = data[~mask]

roll = data[:,:3].astype(np.float32)
phi = data[:,-1]


hole = data_hole[:,:3].astype(np.float32)
phi_hole = data_hole[:,-1]

np.save('data/small_swiss_roll.npy',data)
np.save('data/small_swiss_hole.npy',data_hole)