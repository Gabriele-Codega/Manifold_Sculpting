# Manifold Sculpting
This repository contains a Python implementation of the Manifold Sculpting algorithm, by [M. Ghashler et al.](https://proceedings.neurips.cc/paper/2007/file/c06d06da9666a219db15cf575aff2824-Paper.pdf)

## Motivation
Implement the algorithm and compare it to other dimensionality reduction/manifold learning techniques. This is the final project for the exam in Unsupervised Learning at University of Trieste.

## Dependencies
- numpy
- matplotlib
- numba
Note: numba is not strictly necessary and can be avoided by removing the `@jitclass` decorator. This will massively affect the runtime, making it unbearably slow for (not-so-)large datasets.

## Structure
- ManifoldSculpting: implementation of the algorithm
    - ManifoldSculpting.py: implementation of a class for the Manifold Sculpting technique
- other `.py` or `.ipynb` files are used for testing and need to be cleaned up

## Notes on the implementation
The algorithm requires many iterations over the data points, which for large datasets can take a very long time with python loops. To speed things up, this code makes use of numba's `@jitclass` decorator. To make the code suitable for numba, some parts are not implemented in the most efficient way. For instance:
- The search for nearest-neighbours is done by computing pairwise distances between all data points and could be sped-up by implementing a KDTree algorithm. (Although the search is done once and for all, in an _offline_ phase ad the beginning of the training)
- The queue is simply implemented as a list, which can become slow for large datasets.
- Some loops are not written in the most _pythonic_ way

Note that for the present application, these non-optimal implementations are not detrimental to the performance of the code and the use of numba offers considerable speed-ups.