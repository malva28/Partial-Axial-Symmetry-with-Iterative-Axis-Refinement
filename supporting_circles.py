import warnings

import numpy as np
from progress_bar import print_progress_bar


class Circle:
    def __init__(self, center, radius, normal):
        self.c = center
        self.r = radius
        self.n = normal

    def __str__(self):
        return "Circle = (r: {} ; c: {} ; n: {})".format(self.r, self.c, self.n)

    def __eq__(self, other):
        return (self.c == other.c).all() and self.r == other.r and (self.n == other.n).all()

    def get_phi(self):
        """
        Gets the azimuthal angle of the circle's normal (rotation from the initial meridian plane in spherical coords)
        :return:
        """
        return get_phi_from_normal(self.n)

    def get_theta(self):
        """
        Gets the polar angle of the circle's normal (inclination from respect to the polar axis z)
        :return:
        """
        return get_theta_from_normal(self.n)

    def get_c_r_n_tuple(self) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Gets the center, radius and normal in a 3-tuple
        :return:
        """
        return self.c, self.r, self.n

    def generate_circle_node_edges(self, n_nodes=10, angle_shift: float = 0):
        return generate_circle_node_edges_from_circle_vals(*self.get_c_r_n_tuple(), n_nodes, angle_shift)

    def generate_random_circle_node_edges(self, n_nodes: int = 10):
        return generate_random_circle_node_edges_from_circle_vals(*self.get_c_r_n_tuple(), n_nodes)


def get_phi_from_normal(normal: np.ndarray):
    if normal.ndim == 1:
        x, y, z = normal
        return np.arctan(y / x)
    else:
        return np.arctan2(normal[:,1], normal[:,2])


def get_theta_from_normal(normal: np.ndarray):
    if normal.ndim == 1:
        x, y, z = normal
        return np.arccos(z)
    if normal.ndim == 2:
        return np.arccos(normal[:, 2])

def get_orthonormal_system(a_vec):
    # let v1, v2, n an orthonormal system
    v1 = np.array([a_vec[1], -a_vec[0], 0])
    if np.array_equal(v1, np.zeros(3)):  # n is a Z(+|-) vector, so v1 has to be calculated in another way
        v1 = np.array([1, 0, 0])  # but any (X|Y)(+|-) will do it.

    v2 = np.cross(a_vec, v1)

    # make them orthonormal
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    return v1, v2


def unit_vector_by_polar_coords(unit_coord_1: np.ndarray,
                                unit_coord_2: np.ndarray,
                                origin: np.ndarray,
                                radius: float,
                                angle: float):
    return origin + radius * (unit_coord_1 * np.cos(angle) + unit_coord_2 * np.sin(angle))


def generate_circle_node_edges_from_circle_vals(
        c: np.ndarray,
        r: float,
        n: np.ndarray,
        n_nodes: int = 10,
        angle_shift: float = 0):

    v1, v2 = get_orthonormal_system(n)

    nodes = []
    edges = []
    for i in range(0, n_nodes):
        theta = angle_shift + i * 2 * np.pi / n_nodes
        nodes.append(unit_vector_by_polar_coords(v1, v2, c, r, theta))
        edges.append([i, (i + 1) % n_nodes])

    return np.array(nodes), np.array(edges)


def generate_random_circle_node_edges_from_circle_vals(
        c: np.ndarray,
        r: float,
        n: np.ndarray,
        n_nodes: int = 10,
        angle_shift: float = 0):

    v1, v2 = get_orthonormal_system(n)

    nodes = []
    edges = []
    angles = np.random.uniform(0, 2*np.pi, n_nodes)
    angles = np.sort(angles)
    for i in range(0, n_nodes):
        theta = angles[i]
        nodes.append(unit_vector_by_polar_coords(v1, v2, c, r, theta))
        edges.append([i, (i + 1) % n_nodes])

    return np.array(nodes), np.array(edges)

class SupportingCircle:
    def __init__(self, circle_candidates=500, max_dist_threshold=None):
        """
        Initializes a SupportingCircle object to be able to fit a supporting circle.
        :param max_dist_threshold: Maximum distance to consider a point be part of a circle candidate
        """
        self.supporting_circle = None
        self.votes = 0
        self.n_candidates = circle_candidates
        if max_dist_threshold is None:
            warnings.warn("Warning... You should set a threshold to validate the supporting circles.")
            max_dist_threshold = 0.05
        self.max_dist_threshold = max_dist_threshold

    def get_supporting_circle(self):
        assert self.supporting_circle is not None, "The Supporting Circle hasn't been fitted with a point_set."
        return self.supporting_circle

    def _step(self, point_set):
        pass

    def fit(self, point_set):
        circle_candidates = []
        for _ in range(self.n_candidates):
            # ransac three points of similar_fm_points, generate circles and validate
            rand_indices = np.random.choice(len(point_set), size=3, replace=False)
            qa, qb, qc = point_set[rand_indices]
            circle = self._compute_circle_candidate(qa, qb, qc)
            votes = self._validate_circle(circle, point_set, self.max_dist_threshold)
            circle_candidates.append((circle, votes))

        self.supporting_circle, self.votes = max(circle_candidates, key=lambda c: c[1])  # the candidate with max votes
        return self.supporting_circle, self.votes

    @staticmethod
    def _compute_circle_candidate(a, b, c) -> Circle:
        # vectors ab and ac
        v1 = b - a
        v2 = c - a

        # center has the form a + lambda1*v1 + lambda2*v2
        # so, we find those vectors and scalars:

        v11 = np.dot(v1, v1)
        v12 = np.dot(v1, v2)
        v22 = np.dot(v2, v2)

        common = 2*(v11*v22 - v12**2)
        lambda_1 = v22*(v11-v12) / common
        lambda_2 = v11*(v22-v12) / common

        center = a + lambda_1*v1 + lambda_2*v2
        radius = np.linalg.norm(lambda_1*v1 + lambda_2*v2)  # ||center - a||
        normal = np.cross(v1, v2)
        normal = normal/np.linalg.norm(normal)

        return Circle(center, radius, normal)

    @staticmethod
    def _distance_to_circle(p, circle: Circle):
        c, r, n = circle.get_c_r_n_tuple()

        d = p - c  # vector from the center c to p
        nd = np.dot(n, d)

        if np.linalg.norm(d - n * nd) == 0:  # check if P is over n
            return np.sqrt(r ** 2 + np.linalg.norm(d))

        pq2 = nd ** 2
        kq2 = (np.linalg.norm(np.cross(n, d)) - r) ** 2

        return np.sqrt(pq2 + kq2)

    @staticmethod
    def _validate_circle(circle: Circle, point_cluster, max_dist_threshold):
        votes = 0
        for p in point_cluster:
            if SupportingCircle._distance_to_circle(p, circle) < max_dist_threshold:
                votes += 1

        return votes


def compute_supporting_circles(point_sets, candidates_per_circle, max_dist_threshold):
    """
    For each point set, calculates a supporting circle.
    :param point_sets:
    :param candidates_per_circle: How many candidates calculated for each supporting circle, before selecting one
    :param max_dist_threshold:  The maximum distance to consider a point part of a circle
    :return: A list of supporting circles, as many as point sets
    """
    supporting_circles = []
    votes = []
    _ = 0
    for point_set in point_sets:
        sc = SupportingCircle(circle_candidates=candidates_per_circle,
                              max_dist_threshold=max_dist_threshold)
        s_circle, s_circle_votes = sc.fit(point_set)
        supporting_circles.append(s_circle)
        votes.append(s_circle_votes)
        print_progress_bar(_+1, len(point_sets), prefix='Progress:', length=20)

        _ += 1

    return supporting_circles, votes
