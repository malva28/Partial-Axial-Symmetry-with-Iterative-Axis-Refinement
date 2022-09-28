import random
import numpy as np


class SupportingCircle:
    def __init__(self):
        pass

    def get_supporting_circle(self):
        pass

    def _step(self, point_set):
        pass

    def fit(self, point_set):
        while True:
            # self._step(point_set)

            # ransac three points of similar_fm_points

            qa, qb, qc = random.sample(point_set, 3)
            c, r, n = self._compute_circle_candidate(qa, qb, qc)
            # self._validate_circle((c, r, n), point_set, threshold=0.05)
            pass
        pass

    @staticmethod
    def _compute_circle_candidate(a, b, c):
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

        return center, radius, normal

    @staticmethod
    def _validate_circle(circle, point_cluster, threshold):
        c, r, n = circle
        votes = 0
        for p in point_cluster:
            # if distance_to_circle(p, circle) < threshold:
            #   votes += 1
            pass


def compute_supporting_circles(point_sets):
    for point_set in point_sets:
        sc = SupportingCircle()
        sc.fit(point_set)

    pass
