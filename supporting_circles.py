import numpy as np
from progress_bar import print_progress_bar


class SupportingCircle:
    def __init__(self):
        self.supporting_circle = None
        self.votes = 0

    def get_supporting_circle(self):
        assert self.supporting_circle is not None, "The Supporting Circle hasn't been fitted with a point_set."
        return self.supporting_circle

    def _step(self, point_set):
        pass

    def fit(self, point_set):
        circle_candidates = []
        for _ in range(min(100, len(point_set))):
            # ransac three points of similar_fm_points, generate circles and validate
            rand_indices = np.random.choice(len(point_set), size=3, replace=False)
            qa, qb, qc = point_set[rand_indices]
            circle = self._compute_circle_candidate(qa, qb, qc)
            votes = self._validate_circle(circle, point_set, threshold=0.05)
            circle_candidates.append((circle, votes))

        self.supporting_circle, self.votes = max(circle_candidates, key=lambda c: c[1])  # the candidate with max votes
        return self.supporting_circle, self.votes

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
        normal = normal/np.linalg.norm(normal)

        return center, radius, normal

    @staticmethod
    def _distance_to_circle(p, circle):
        c, r, n = circle

        d = p - c  # vector from the center c to p
        nd = np.dot(n, d)

        if np.linalg.norm(d - n * nd) == 0:  # check if P is over n
            return np.sqrt(r ** 2 + np.linalg.norm(d))

        pq2 = nd ** 2
        kq2 = (np.linalg.norm(np.cross(n, d)) - r) ** 2

        return np.sqrt(pq2 + kq2)

    @staticmethod
    def _validate_circle(circle, point_cluster, threshold):
        votes = 0
        for p in point_cluster:
            if SupportingCircle._distance_to_circle(p, circle) < threshold:
                votes += 1

        return votes


def compute_supporting_circles(point_sets):
    supporting_circles = []
    _ = 0
    for point_set in point_sets:
        sc = SupportingCircle()
        supporting_circles.append(sc.fit(point_set))
        print_progress_bar(_+1, len(point_sets), prefix='Progress:', length=20)

        _ += 1

    return supporting_circles
