import numpy as np
from scipy.spatial.distance import pdist, squareform


class ClusterPoint:
    def __init__(self):
        self.descriptor = []
        self.idx = None
        self.cla = None


class Cluster:
    def __init__(self, dim=0):
        self.dim = dim
        self.centroid = ClusterPoint()
        self.points = []

        self.centroid_idx = -1
        self.members = []

    def compute_centroid(self):

        self.centroid.descriptor.clear()
        self.centroid.descriptor = [0]*self.dim

        for i in range(len(self.points)):
            for j in range(self.dim):
                self.centroid.descriptor[j] += self.points[i].descriptor[j]

        for j in range(self.dim):
            self.centroid.descriptor[j] /= len(self.points)

    def compute_medoid(self, distances, n):
        factor = np.inf
        pos = -1

        for i in range(len(self.members)):
            s = 0
            for j in range(len(self.members)):
                s += distances[self.members[i]][self.members[j]]  # self.members[i]*n + self.members[j]

            if s < factor:
                factor = s
                pos = i

        self.centroid_idx = self.members[pos]

    def add_point(self, p: ClusterPoint):
        self.points.append(p)

    def add_idx(self, idx: int):
        self.members.append(idx)

    def remove_points(self, points: list[ClusterPoint]):
        for i in range(len(self.points)):
            points.append(self.points[i])

    def remove_indices(self, marks: list[int]):
        for i in range(len(self.members)):
            marks[self.members[i]] = 0

    def get_size(self):
        return len(self.points)

    def get_size_index(self):
        """
        :return: the number of elements in this cluster
        """
        return len(self.members)

    def get_index_at(self, pos):
        return self.points[pos].idx

    def get_index(self, i):
        """
        Gets the index (in the original set of un-clustered elements) of the i-th element of this cluster
        :param i: to look for the i-th element of this cluster
        :return: the index in the original set of un-clustered elements
        """
        return self.members[i]

    def get_closer(self):
        idx = -1
        min_dist = np.inf

        for i in range(len(self.points)):
            dist = self.compute_distance(self.points[i], self.centroid)
            if dist < min_dist:
                min_dist = dist
                idx = self.points[i].idx

    @staticmethod
    def compute_distance(a: ClusterPoint, b: ClusterPoint):
        s = 0
        for i in range(len(a.descriptor)):
            s += (a.descriptor[i] - b.descriptor[i]) ** 2

        return np.sqrt(s)


def angular_distance(circle_1, circle_2):
    _, _, n1 = circle_1
    _, _, n2 = circle_2

    return 1 - np.abs(np.dot(n1, n2)) / np.linalg.norm(n1)*np.linalg.norm(n2)


def axial_distance(circle_1, circle_2):
    c1, _, n1 = circle_1
    c2, _, n2 = circle_2

    c_dist = c1 - c2
    return np.linalg.norm(np.cross(n1, c_dist)) + np.linalg.norm(np.cross(n2, c_dist))


def adaptive_clustering_medoids(distances, n_points, min_cluster_dist, max_spread, min_cluster_elements):
    clusters = []

    mark = [0] * n_points

    Iter = 10
    for _ in range(Iter):
        deleted = []

        # for each point p
        for i in range(n_points):
            if mark[i]:
                continue

            min_dist = np.inf

            # Find the nearest cluster k to the i-th point
            nearest_cluster_idx = None
            for j in range(len(clusters)):
                centroid_idx = clusters[j].centroid_idx
                dist = distances[centroid_idx][i]  # centroid_idx * n_points + i
                if dist < min_dist:
                    min_dist = dist
                    nearest_cluster_idx = j

            if min_dist == np.inf or min_dist >= max_spread:
                clu = Cluster()
                clu.add_idx(i)
                clu.compute_medoid(distances, n_points)
                clusters.append(clu)
                deleted.append(i)
            elif min_dist <= min_cluster_dist:
                clusters[nearest_cluster_idx].add_idx(i)
                deleted.append(i)
            else:
                print("Else")

        print("Iter:", _, "-> Num. clusters:", len(clusters))

        deleted.reverse()
        for deleted_idx in deleted:
            mark[deleted_idx] = 1
        deleted.clear()

        for i in range(len(clusters)):
            if clusters[i].get_size_index() >= min_cluster_elements:
                clusters[i].compute_medoid(distances, n_points)
            else:
                clusters[i].remove_indices(mark)
                deleted.append(i)

        deleted.reverse()
        for deleted_idx in deleted:
            clusters.pop(deleted_idx)

    return clusters


def get_circles_of_most_populated_cluster(circles, clusters):
    """
    Gets the circles of the most populated cluster.
    :param circles: Universe of circles
    :param clusters: List of clusters of the circles
    :return:
    """

    # get the cluster with more elements
    max_cluster = clusters[0]
    counts = []
    for cluster in clusters:
        counts.append(cluster.get_size_index())
        if max_cluster.get_size_index() < cluster.get_size_index():
            max_cluster = cluster
    print(counts)
    # retrieve its circles
    max_cluster_circles = []
    for idx in max_cluster.members:
        max_cluster_circles.append(circles[idx])

    return max_cluster_circles


def compute_generator_axis(circles):
    # cluster the circles by angular distance
    angular_dists = squareform(pdist(circles, metric=angular_distance))
    similar_angle_clusters = adaptive_clustering_medoids(angular_dists, len(circles), 0.03, 0.015, 10)
    max_cluster_circles = get_circles_of_most_populated_cluster(circles, similar_angle_clusters)

    # cluster the selected circles by axial distance
    axial_dists = squareform(pdist(max_cluster_circles, metric=axial_distance))
    similar_axis_clusters = adaptive_clustering_medoids(axial_dists, len(max_cluster_circles), 0.01, 0.005, 10)
    pass
