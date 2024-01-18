# Overview of Manifold Sculpting

_This is a collection of thoughts about this method, mainly because it helps me understand it. This is not very polished, so be aware of that shall you decide to read this._

## MS and Nonmetric MDS
The idea is pretty simple and in general is not so different from other manifold learning techniques. In fact, the loss function for MS is possibly strictly related to a generic loss from nonmetric MDS. 

Indeed, the loss from MS is ultimately a function of the distance between points, only it considers distances between neighbours instead of all the distances. The dependency on the distance is clear in the first term of the loss, while for the second term one can make this dependence explicit by noting that
$$ \theta_{ij} = \arccos(\frac{x_i\cdot x_j}{||x_i||~||x_j||}) \\
= \arccos(\frac{G_{ij}}{(G_{ii}~G_{jj})^{1/2}}) $$
where $G$ is the Gram matrix. One can then drop the $\arccos$ and rewrite $G = -\frac{1}{2} C~D~C$ where $C$ is the centering matrix and $D$ is the distance matrix. 
Note that if one disposes of the $arccos$, then the normalisation may change in some other way.

After all this, it may be possible to rewrite the loss in the form 
$$ e = \sum_{ij} f(D) - g(D_0)$$
where $D_0$ is the distance matrix for the original data. This form is somewhat analogous to the stress on nonmetric MDS.

## Parameters
In principle, this technique only has two parameters: $k$, the number of neighbours, and $\sigma$, the squishing rat. 

I believe that the choice of $k$ is somewhat easier as it relates to the size of the neighbourhood where we want local relationships to be preserved, hence one can make some educated guess on the appropriate $k$ based on how _flat_ the manifold is. In all fairness, this is not so easy as the dimensionality of data grows beyond 3.

The choice of $\sigma$ is somewhat less intuitive, at least to me, as it is not so easy to decide which is an appropriate squishing rate of unwanted dimensions.

To add insult to injury, it seems that the two parameters are somewhat related to eachother, and a change in $k$ might then require an appropriate change in $\sigma$ to get good results, or vice versa.

When implementig the actual algorithm, moreover, one is faced with the choice of more parameters, that also influence the quality of the solution and the speed of convergence.

The first parameter is the learning rate, or step, for the hill climbing process in the minimisation step. Indeed one needs to decide a proper step size to adjust coordinates, as a very large step may move the data all over the place without any convergence to a minimum whatsoever, and a very small stepsize might cause the system to fall into a local minimum and the processing time could become unbearably long.

The solution to these problems is not clearly addressed in the paper, although it seems a pretty critical issue to me, but can be found deep into the source code of Waffles. The best learning rate to start with is apparently the average distance between neighbours, which a posteriori seems very reasonable. However, this initial value is not only rescaled by a random amount in $[0.6,1)$ for each point, but is also globally scaled down by a factor $0.87$ or scaled up by a factor $0.91$ (i.e. `lr *= 0.87` or `lr /= 0.91`) depending on how many hill climbing steps are performed at each iteration. This is once again reasonable, as it helps the convergence to minima and speeds up the process when the system is far from the minimum. Nonetheless, I have no idea about how they figured out these values to adjust the learning rate.

The second "parameter" is the stopping criterion. It could be a condition on the maximum number of iterations, on the change in error or on the change of coordinates. In the paper they suggest the third but in the code they implement the second. Not only do they implement the second, they also set yet another parameter, which could be called _patience_, that determines a maximum amount of iterations that the program should wait after not seeing any improvements in the error. This seems a good criterion, although one now needs to take extra care in at least two other things:

1. do a certain amount (again, one more parameter) of _burn in_ iterations to get the system into a state where the error is large enough, hence suitable for comparison. This is necessary as the error in the first _n_ iterations will be very small, possibly smaller than it will ever be in subsequent iterations. This means that if we were to start comparing errors from the beginning, the algorithm would stop almost immediately, as the number of iterations before the error starts to decrease again is usually much larger than the patience.
2. deal with the fact that the system could fall into a local minimum and then try to (slowly) get out of it. This is an issue because if the number of iterations to get out of the local minimum is larger than the patience, the final state of the system will be significantly worse than the one at the local minimum (possibly close to a local maximum instead), hence the data after the final iteration will be terrible.

**The point is that this algorithm is hard to tune.**

In all fairness, though, other algorithms are pretty hard to tune as well and the problem of the number of _actual_ parameters one needs to set in the actual implementation is somewhat common to many other algorithms.

For instance, t-SNE has one important parameter, parplexity, but the sklearn implementation of t-SNE accepts a bunch more parameters. The same is true for UMAP, which has the number of neighbours and a minimum distance as parameters, but in practice accepst a bunch more in the sklearn implementation. ISOMAP, on the other hand, works just fine with just the number of neighbours.

## Why MS?
In the paper, the authors claim that MS is better than some others manifold learning techniques, as it is more accurate and at times faster. In the case of the swiss roll I could not see much difference in the results, and frankly my results did not quite match those from the paper, but still it is indeed very possible that this algorithms outperforms others in real-life scenarios, where data are more complex.

My main concern is that ISOMAP works pretty well and pretty fast on this data, while MS requires tuning and is slower. It's possible that ISOMAP is fast because the dataset is small, and the complexity of dense linear algebra does not matter on this size, especially since my implementation of MS is most definitely not optimised and relies heavily of possibly very slow for loops.

Another observation is that in the paper they compare quality of algorithms based on the error with respect to the analytical parametrisation of the manifold. I am not sure whether this is necessarily a good metric, and I would argue that it depends on the specific application. For instance, if I was to employ one of the methods in the pipeline for a classification model, I probably would care more about the accuracy of the classification than the accuracy of the embedding with respect to the actual parametrisation. 

Indeed when they show an example on images and videos, ISOMAP ultimately performs very nicely, pretty much like MS.

## Possible extensions
The strength of this method seems to be the loss, that not only aims to preserve distances but also point alignment. So I wonder if it could be possible to use the loss to determine a parametric mapping, which could also be applied to out of sample points. 

Maybe this is possible to do with a neural network, but at this point I still have not managed to make it work, so I have no idea whether it would be useful or not.