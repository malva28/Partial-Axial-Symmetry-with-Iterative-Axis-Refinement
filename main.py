import sys
import subprocess
import openmesh

from common_args import create_common_args, symmetry_param_keys, add_shift_args, shift_params_keys
from mesh_axis_render import show_mesh_with_partial_axis
from utils import cache_exists, cache_empty, read_cache, find_row_in_cache, get_generator_circle_from_cache
from iterative_shift import iterative_symmetry_shift, compute_normal_angles_change, plot_loss_and_angle_change, \
    save_normal_angle_diff_into_csv, convert_nlad_array_into_dict
from transformations import normalize, reorient_point_cloud_by_angles


def get_row_and_angles_from_mesh(mesh, **parser_vals):
    df = read_cache()

    row, _ = find_row_in_cache(df, **parser_vals)

    phi_adjust = row.iloc[0]["phi_adjust"]
    theta_adjust = row.iloc[0]["theta_adjust"]
    return phi_adjust, theta_adjust, row


def use_from_cache(mesh, visual, **parser_vals):
    print("Using cached values for symmetry axis")
    phi_adjust, theta_adjust, row = get_row_and_angles_from_mesh(mesh, **parser_vals)
    generator_circle = get_generator_circle_from_cache(row)
    print(generator_circle)

    # Reorientation
    reorient_point_cloud_by_angles(mesh.points(), phi_adjust, theta_adjust)

    if visual:
        show_mesh_with_partial_axis(
            mesh,
            generator_circle,
            args.symmetric_support_threshold,
            phi_adjust,
            theta_adjust,
            symmetric_support=False)
    return generator_circle


def run_initial_partial_symmetry():
    venv_exe = sys.executable
    flat_args = []

    key_args = symmetry_param_keys.copy()
    key_args.extend(["visual"])

    for key in key_args:
        value = arg_dict[key]
        if type(value) == bool:
            # flag found
            if value:
                flat_args.append("--" + key)
            elif not value and key == "visual":
                flat_args.append("--no-visual")
        else:
            flat_args.append("--" + key)
            flat_args.append(str(value))
    process = subprocess.run([venv_exe, 'main_partial_symmetry.py', *flat_args], check=True)
    ret = process.returncode
    print("Finished executing main_partial_symmetry with return code {}".format(ret))
    return ret


if __name__ == "__main__":
    parser = create_common_args(description='Mesh signature visualization')
    add_shift_args(parser)
    parser.add_argument('--cache', default=True, action='store_true', help="True if you want to retrieve initial axis "
                                                                           "from cache file.\nFalse if you want to "
                                                                           "reforce partial axis calculation")
    parser.add_argument('--no-cache', dest='cache', action='store_false')

    args = parser.parse_args()
    arg_dict = vars(args)
    parser_vals = {key: arg_dict[key] for key in symmetry_param_keys}

    used_args = {}
    for key in shift_params_keys:
        used_args[key] = getattr(args, key)

    print("Reading Mesh")
    mesh = openmesh.read_trimesh(args.file)
    normalize(mesh.points())

    not_cached = False

    if args.cache:
        exists_ = cache_exists()
        not_cached = not cache_exists() or cache_empty() or find_row_in_cache(read_cache(), **parser_vals)[0].empty

    not_cached = not args.cache or not_cached

    if not_cached:
        ret = run_initial_partial_symmetry()
        df = read_cache()
        row, _ = find_row_in_cache(df, **parser_vals)
        generator_circle = get_generator_circle_from_cache(row)
    else:
        generator_circle = use_from_cache(mesh, args.visual, **parser_vals)

    best_normals, new_generator_circle, iters, additional = iterative_symmetry_shift(mesh,
                                                                  generator_circle,
                                                                  args.delta_radius,
                                                                  args.decrease_factor,
                                                                  args.phi_simmetries,
                                                                  args.epsilon_radius,
                                                                  args.chi_convergence,
                                                                  args.n_sample_points,
                                                                  args.n_angles)
    for key in additional:
        used_args[key] = additional[key]

    if args.visual:
        show_mesh_with_partial_axis(
            mesh,
            new_generator_circle,
            args.symmetric_support_threshold,
            phi=0.0,
            theta=0.0,
            symmetric_support=False)

    best_normals_with_angle_diff = compute_normal_angles_change(best_normals, generator_circle)
    #plot_loss_and_angle_change(best_normals_with_angle_diff)
    data, n_rows = convert_nlad_array_into_dict(best_normals_with_angle_diff, iters)

    save_normal_angle_diff_into_csv(args.file, n_rows, data, **used_args)
