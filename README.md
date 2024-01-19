# Manifold Sculpting
This repository contains a Python implementation of the Manifold Sculpting algorithm, as described by [M. Ghashler et al.](https://proceedings.neurips.cc/paper/2007/file/c06d06da9666a219db15cf575aff2824-Paper.pdf)
There are some tricks to the implementation which are not reported in the paper but can be found in the source code of [Waffles](https://github.com/mikegashler/waffles/tree/master).

## Motivation
Implement the algorithm and compare it to other dimensionality reduction/manifold learning techniques. This is the final project for the exam in Unsupervised Learning at University of Trieste.

## Dependencies
- numpy
- matplotlib
- numba
Note: numba is not strictly necessary and can be avoided by removing the `@jitclass` decorator. This will massively affect the runtime, making it unbearably slow for (not-so-)large datasets.

## Structure
- `ManifoldSculpting`: implementation of the algorithm
    - ManifoldSculpting.py: implementation of a class for the Manifold Sculpting technique
- `run.py`: example of how one could use the class
- `generate_data.py`: generate the swiss_roll and swiss_hole datasets
- `other`: implementation of other techniques
    - `run_others.py`: implementation of isomap and kernel PCA and transformation of the data using
    isomap, kernel PCA, t-SNE, UMAP.
    - `run_autoencoder.py`: attempt to apply an autoencoder to the swiss roll dataset. Parameters for
    a trained model are available in `autoencoder.pt` but the 2D representation it gives is not very pretty (although the reconstruction error is not bad)
- `data`: data from original manifold and various embeddings
- `figs`: plots from some runs of the algorithms
- `slides.*`: slides for a short presentation (the html looks nicer but possibly won't render).

## Notes on the implementation
The algorithm requires many iterations over the data points, which for large datasets can take a very long time with python loops. To speed things up, this code makes use of numba's `@jitclass` decorator. To make the code suitable for numba, some parts are not implemented in the most efficient way. For instance:
- The search for nearest-neighbours is done by computing pairwise distances between all data points and could be sped-up by implementing a KDTree algorithm. (Although the search is done once and for all, in an _offline_ phase ad the beginning of the training)
- The queue is simply implemented as a list, which can become slow for large datasets.
- Some loops are not written in the most _pythonic_ way

Note that for the present application, these non-optimal implementations are not detrimental to the performance of the code and the use of numba offers considerable speed-ups.
