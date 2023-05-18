import argparse
import laplace
import signature

symmetry_params = {
    "file": dict(
        default='files/cat0.off',
        type=str,
        help='File to use'),
    "n_basis": dict(
        default='100',
        type=int,
        help='Number of basis used'),
    "approx": dict(
        default='cotangens',
        choices=laplace.approx_methods(),
        type=str,
        help='Laplace approximation to use'),
    "signature": dict(
        default='heat',
        choices=signature.kernel_signatures(),
        type=str,
        help='Kernel signature to use'),
    # FPS-related
    "n_samples": dict(
        default=30,
        type=int,
        help='number of points to sample with FPS'),
    # Supporting Circles-related
    "n_candidates_per_circle": dict(
        default=500,
        type=int,
        help='Number of circle candidates from where to choose each single Supporting Circle.'),
    "circle_candidate_threshold": dict(
        default=0.005,
        type=float,
        help='Threshold to consider a point part of a circle candidate. Distance to circle candidate '
             '(float representing percentage of diagonal)'),
    # Clustering-related (Generator Axis)
    "angular_r": dict(
        default=0.015,
        type=float,
        help='Maximum distance point-centroid in the angular-clustering.'),
    "angular_s": dict(
        default=0.03,
        type=float,
        help='Minimum distance to a distant point in the angular-clustering.'),
    "angular_k": dict(
        default=10,
        type=int,
        help='Minimum number of points per cluster in the angular-clustering.'),
    "axis_r": dict(
        default=0.25,
        type=float,
        help='Maximum distance point-centroid in the axis-clustering.'),
    "axis_s": dict(
        default=0.5,
        type=float,
        help='Minimum distance to a distant point in the axis-clustering.'),
    "axis_k": dict(
        default=5,
        type=int,
        help='Minimum number of points per cluster in the axis-clustering.'),
    # Symmetric Support-related
    "symmetric_support_threshold": dict(
        default=0.01,
        type=float,
        help='Threshold to consider a point affected by axial symmetry. Distance to the axial circle. '
             '(float representing percentage of diagonal)')
}

symmetry_param_keys = list(symmetry_params.keys())


def create_common_args(description):
    parser = argparse.ArgumentParser(description='Mesh signature visualization')
    for key in symmetry_param_keys:
        param_dict = symmetry_params[key]
        parser.add_argument("--" + key, **param_dict)
    # Visual related
    parser.add_argument('--visual', default=True, action='store_true', help="True if you want to display results.")
    parser.add_argument('--no-visual', dest='visual', action='store_false')

    return parser
