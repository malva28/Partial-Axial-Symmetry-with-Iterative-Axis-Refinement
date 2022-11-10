import numpy as np
from scipy.spatial.distance import pdist, squareform
from supporting_circles import Circle


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
        """
        Unmark the cluster's points from a list of marked points,
        using the original index in the set of un-clustered elements.

        :param marks: A marking list, with length of the number of elements in the un-clustered set
        """
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


def adaptive_clustering_medoids(distances, n_points, r, s, min_cluster_elements):
    """
    Customized clustering algorithm. Uses a matrix distance between points to form clusters.
    Points near enough to an existing cluster will be considered part of it.
    Points far enough to the nearest cluster will create a new cluster.
    Points with neither of those conditions will be not part of any cluster, discarding possible noise between clusters.

    Runs this algorithm a few times over the un-clustered points, and discarding clusters with few elements.

    :param distances: Distance-matrix between the un-clustered n_points
    :param n_points: Number of un-clustered points to clusterize
    :param r: Maximum distance point-centroid to be considered part of a cluster
    :param s: Minimum distance to a distant point to form a new cluster.
    :param min_cluster_elements: Minimum number of points that a cluster must have to be considered a proper cluster.
    :return: The set of clusters
    """
    clusters = []

    mark = [0] * n_points

    Iter = 10
    for _ in range(Iter):

        # for each point p
        for i in range(n_points):
            if mark[i]:  # Ignore points that are already in clusters
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

            if min_dist >= s:
                # Point too far away of the closest cluster
                # Create a new cluster and remove the point (adds it to the deleted list, to remove it later)
                clu = Cluster()
                clu.add_idx(i)
                clu.compute_medoid(distances, n_points)
                clusters.append(clu)
                mark[i] = 1
            elif min_dist <= r:
                # Point close enough to be considered part of the cluster
                # Append it to the cluster, and remove it from
                clusters[nearest_cluster_idx].add_idx(i)
                mark[i] = 1
            else:
                # between r and s, the point is not close enough to be part of a cluster,
                # but not too far away to form a new one
                print("Else")

        delete_mark = [0] * len(clusters)
        # Validate clusters
        for i in range(len(clusters)):
            if clusters[i].get_size_index() >= min_cluster_elements:
                clusters[i].compute_medoid(distances, n_points)
            else:
                # Not enough points in this cluster, unmark its points and remove the cluster
                clusters[i].remove_indices(mark)
                delete_mark[i] = 1

        print("Iter:", _, "-> Num. clusters:", len(clusters), ", removing", delete_mark.count(1))
        # Filter the deleted clusters
        clusters = [clusters[i] for i in range(len(clusters)) if not delete_mark[i]]

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
    # print(counts)
    # retrieve its circles
    max_cluster_circles = []
    for idx in max_cluster.members:
        max_cluster_circles.append(circles[idx])

    return max_cluster_circles


def circle_average(circles: list[Circle]):
    c_avg, r_avg, n_avg = np.array([0, 0, 0], dtype=float), 0, np.array([0, 0, 0], dtype=float)
    for circle in circles:
        c, r, n = circle.get_c_r_n_tuple()
        c_avg += c
        r_avg += r
        # simply adding same-orientation different-sense vectors result in cancelling important vectors
        # so, we make them all positive in x-axis
        n_avg += n if n[0] >= 0 else -n

    c_avg /= len(circles)
    r_avg /= len(circles)
    n_avg /= np.linalg.norm(n_avg)
    return Circle(c_avg, r_avg, n_avg)


def compute_generator_axis(circles: list[Circle]):
    # transform the 1-D list of circles into a 2-D array
    circles = np.array([circle.get_c_r_n_tuple() for circle in circles])

    # cluster the circles by angular distance
    angular_dists = squareform(pdist(circles, metric=angular_distance))
    similar_angle_clusters = adaptive_clustering_medoids(angular_dists, len(circles), 0.03, 0.015, 10)
    max_cluster_circles = get_circles_of_most_populated_cluster(circles, similar_angle_clusters)

    # cluster the selected circles by axial distance
    axial_dists = squareform(pdist(max_cluster_circles, metric=axial_distance))
    print(axial_dists)
    similar_axis_clusters = adaptive_clustering_medoids(axial_dists, len(max_cluster_circles), 0.5, 0.25, 5)
    max_cluster_circles = get_circles_of_most_populated_cluster(max_cluster_circles, similar_axis_clusters)

    # transform back the 2-D array into a 1-D list of circles
    max_cluster_circles = [Circle(circle[0], circle[1], circle[2]) for circle in max_cluster_circles]
    return max_cluster_circles, circle_average(max_cluster_circles)  # WIP, should only return the circle_average
