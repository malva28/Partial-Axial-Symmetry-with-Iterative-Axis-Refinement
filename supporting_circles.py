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
            pass
        pass

    @staticmethod
    def _compute_circle_candidate(a, b, c):

        center = 0
        radius = 0
        normal = 0

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
