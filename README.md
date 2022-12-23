# Partial Axial Symmetry

This repository is an implementation and frontend application of the algorithm proposed in
_“Analysis of partial axial symmetry on 3d surfaces and its application in the restoration of cultural heritage objects,”_
(Sipiran, I., 2017).

The goal of this repo is to provide an easy way to experiment with the proposed method, so it's an implementation
in python to the algorithm. The paper develops an interesting way to find axial symmetry from partial 3D objects that are
missing in geometry.

## How to use

Run `python main.py --help` to see the list of parameters from the CLI. The parameters control different parts of the
algorithm, and they are described in the following sections.

```
--file              
                    File to use
```

### Descriptors computation and signatures related

```
--n_basis          
                    Number of basis used
--approx {beltrami,cotangens,mesh,fem,robust}
                    Laplace approximation to use
--signature {heat,wave,global}
                    Kernel signature to use
```

When running with a non-manifold file, or an object with missing geometry, some approximations used may fail and get 
NaN values. You can use the robust-laplacian approximation with `--aprox robust` to solve that problem.

### FPS related

You can propose a number of points to sample with FPS,
but the algorithm will use at least 0.5% of the point cloud. 

```
--n_samples
                    number of points to sample with FPS
```

### Supporting Circles related

The algorithm gets a supporting circle for each sampled points. It uses RANSAC from a set of points that have similar
descriptor values to get a good circle candidate, but this has to be done many times to select the best circle candidate.
The following parameters control how many times is RANSAC ran for each set of points, and how near the points have to be
to the circle candidate to be considered part of the circle.  

```
--n_candidates_per_circle
                    Number of circle candidates from where to choose each single Supporting Circle.
--circle_candidate_threshold
                    Threshold to consider a point part of a circle candidate. Distance to circle candidate (float
                    representing percentage of diagonal)
```

### Generator Axis

The algorithm use clustering to find the supporting circles that have similar axis, to compute the object's axis from them.
The following parameters control both of the clustering process: The angle clustering and the axis clustering.

```
--angular_r
                    Maximum distance point-centroid in the angular-clustering.
--angular_s
                    Minimum distance to a distant point in the angular-clustering.
--angular_k
                    Minimum number of points per cluster in the angular-clustering.
--axis_r
                    Maximum distance point-centroid in the axis-clustering.
--axis_s
                    Minimum distance to a distant point in the axis-clustering.
--axis_k
                    Minimum number of points per cluster in the axis-clustering.
```

### Symmetric Support

The algorithm checks for every point how much of axial symmetry they have, by counting how many of the other points are
axial-symmetric to them. This means we have to use a threshold to consider a point symmetric-enough, and the following
parameter is in charge of that.

```
--symmetric_support_threshold
                    Threshold to consider a point affected by axial symmetry. Distance to the axial circle. (float
                    representing percentage of diagonal)
```

### Non-algorithm related

Sometimes, we might only want to run the algorithm without visualizing the results in the GUI, so these flags allows the
program to end when the computation process finishes.

```
--no-visual
```


## Some references used when developing the algorithm that you may find useful

* Circle from three points: https://github.com/sergarrido/random/tree/master/circle3d
* Distance point-circle: https://www.geometrictools.com/Documentation/DistanceToCircle3.pdf
* Distances https://stackoverflow.com/questions/31667070/max-distance-between-2-points-in-a-data-set-and-identifying-the-points

And of course, the main paper and inspiration to this project,
>Sipiran, I., “Analysis of partial axial symmetry on 3d surfaces and its application in the restoration of cultural heritage objects,” en 2017 IEEE International Conference on Computer Vision
Workshops (ICCVW), pp. 2925–2933, 2017, doi:10.1109/ICCVW.2017.345.