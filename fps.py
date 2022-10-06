# https://github.com/ziruiw-dev/farthest-point-sampling

import os
import numpy as np

from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform
import openmesh

from progress_bar import print_progress_bar


class FPS:
    def __init__(self, pcd_xyz, n_samples):
        self.n_samples = n_samples
        # self.pcd_xyz = pcd_xyz
        self.n_pts = pcd_xyz.shape[0]
        self.dim = pcd_xyz.shape[1]
        self.selected_pts = None
        self.selected_pts_indices = np.zeros(self.n_samples, dtype=int)
        self.selected_pts_expanded = np.zeros(shape=(n_samples, 1, self.dim))
        self.remaining_pts = np.copy(pcd_xyz)

        self.dist_pts_to_selected = None  # Iteratively updated in step(). Finally re-used in group()

        # Random pick a start
        self.start_idx = np.random.randint(low=0, high=self.n_pts - 1)
        self.selected_pts_expanded[0] = self.remaining_pts[self.start_idx]
        self.selected_pts_indices[0] = self.start_idx
        self.n_selected_pts = 1

    def get_selected_pts(self):
        self.selected_pts = np.squeeze(self.selected_pts_expanded, axis=1)
        return self.selected_pts, self.selected_pts_indices

    def _step(self):
        if self.n_selected_pts < self.n_samples:
            self.dist_pts_to_selected = self.__distance__(self.remaining_pts,
                                                          self.selected_pts_expanded[:self.n_selected_pts]).T
            dist_pts_to_selected_min = np.min(self.dist_pts_to_selected, axis=1, keepdims=True)
            res_selected_idx = np.argmax(dist_pts_to_selected_min)

            self.selected_pts_indices[self.n_selected_pts] = int(res_selected_idx)
            self.selected_pts_expanded[self.n_selected_pts] = self.remaining_pts[res_selected_idx]
            self.n_selected_pts += 1
        else:
            print("Got enough number samples")

    def fit(self):
        for _ in range(1, self.n_samples):
            self._step()
            print_progress_bar(_, self.n_samples-1, prefix='Progress:', length=20)
        return self.get_selected_pts()

    @staticmethod
    def __distance__(a, b):
        return np.linalg.norm(a - b, ord=2, axis=2)


def compute_fps(filename, n_samples: int, pc=None):
    name = os.path.splitext(filename)[0]
    if os.path.exists(str(name) + "fp_sampled" + str(n_samples) + '.npz'):
        data = np.load(str(name) + "fp_sampled" + str(n_samples) + '.npz')
        return data['points'], data['indices']

    if pc is None:
        pc = openmesh.read_trimesh(filename).points()
    sample_fps = FPS(pc, n_samples)
    print("Initialised FPS sampler successfully.")
    print("Running FPS over {0:d} points and geting {1:d} samples.".format(pc.shape[0],
                                                                           n_samples))
    sample, sample_idx = sample_fps.fit()  # Get all samples.
    print("FPS sampling finished.")
    np.savez_compressed(str(name) + "fp_sampled" + str(n_samples) + '.npz', points=sample, indices=sample_idx)

    return sample, sample_idx


def farthest_distance(points):
    """
    Computes the convexhull in O(nlogn), then finds the best pair in O(H^2), with H the number of points in the hull
    :param points:
    :return: the max distance between the points
    """
    hull = ConvexHull(points)

    # Extract the points forming the hull
    hull_points = points[hull.vertices, :]

    # Get distance between all points in the hull and get the farthest distance
    hull_distances = squareform(pdist(hull_points, metric='euclidean'))

    # if we wanted to retrieve the indices of the pair:
    # row, col = np.unravel_index(hull_distances.argmax(), hull_distances.shape)
    # return hull_points[row], hull_points[col]

    return np.max(hull_distances)

