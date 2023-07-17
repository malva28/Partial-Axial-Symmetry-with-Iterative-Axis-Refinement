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


shift_params = {
    "delta_radius": dict(
        default=0.268,
        type=float,
        help="Starting radius around the normal endpoint from which new normals are spawned from its arc. \n"
             "Also consider that new normals will be at an angle arctan(delta_radius) with respect to the previous "
             "symmetry axis found."),
    "decrease_factor": dict(
        default=0.7,
        type=float,
        help="Factor < 1 by which is decreased delta radius each time a normal candidate is not found on a given "
             "iteration"
    ),
    "phi_simmetries": dict(
        default=16,
        type=int,
        help="Number of random normals spawned from the delta radius circle"),
    "epsilon_radius": dict(
        default=1e-3,
        type=float,
        help="if delta_radius < epsilon_radius, the search for a new normal concludes"
    ),
    "chi_convergence": dict(
        default=1e-3,
        type=float,
        help="Convergence is the differences in loss values between iterations. If it's less than chi_convergence**2, "
             "then the search for a new normal concludes"),
    "n_sample_points": dict(
        default=1000,
        type=int,
        help="Number of sample points chosen to compute a faster Chamfer Distance. Performs uniform sampling"),
    "n_angles": dict(
        default=6,
        type=int,
        help="Number of equidistant angles in the space [0, 2pi] to be taken to compute the Chamfer Distance"
    )
}

shift_params_keys = list(shift_params.keys())


def create_common_args(description):
    parser = argparse.ArgumentParser(description='Mesh signature visualization')
    for key in symmetry_param_keys:
        param_dict = symmetry_params[key]
        parser.add_argument("--" + key, **param_dict)
    # Visual related
    parser.add_argument('--visual', default=True, action='store_true', help="True if you want to display results.")
    parser.add_argument('--no-visual', dest='visual', action='store_false')

    return parser


def add_shift_args(a_parser: argparse.ArgumentParser):
    for key in shift_params_keys:
        shift_dict = shift_params[key]
        a_parser.add_argument("--" + key, **shift_dict)
    return a_parser
