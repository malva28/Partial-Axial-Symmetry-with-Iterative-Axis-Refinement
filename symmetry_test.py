import argparse
import re
import numpy as np
import openmesh

from common_args import create_common_args, add_shift_args, shift_params_keys
from compute_chamfer import compute_symmetry_chamfer_distance, compute_symmetry_chamfer_distance_from_mesh
from supporting_circles import Circle
from iterative_shift import iterative_symmetry_shift, compute_normal_angles_change, plot_loss_and_angle_change, \
    save_normal_angle_diff_into_csv, show_mesh_with_all_found_axes
from transformations import myangle


def parse_axis_perturbation_file(file_path):
    float_pattern = r"[-]?\d+([.]\d+)?([Ee][-+]?\d+)?"
    xyz_pattern = r"\s".join([r"(?P<{}>{})".format(coord, float_pattern) for coord in ["x", "y", "z"]])
    angle_xyz_pattern = r"(?P<angle>\d+)\s" + xyz_pattern
    with open(file_path, "r") as fopen:
        lines = fopen.readlines()
        res_center = re.match(xyz_pattern, lines[0])
        center = np.array([float(coord) for coord in res_center.groupdict().values()])
        res_first_normal = re.match(xyz_pattern, lines[1])
        normals = [np.array([float(coord) for coord in res_first_normal.groupdict().values()])]
        angles = [0]
        for i in range(2, len(lines)):
            res_variation = re.match(angle_xyz_pattern, lines[i])
            list_variation = list(res_variation.groupdict().values())
            normals.append(np.array([float(val) for val in list_variation[1:]]))
            angles.append(int(list_variation[0]))
    return center, np.stack(normals), np.array(angles)


def run_sanity_test():
    """
    The initial delta radius will be equal to the angle the normal was shifted.
    This way you can check if the algorithm is working with the proper angle subdivisions
    and such, as, theoretically, it should instantly find the original (and better) symmetry axis
    """


def convert_test_array_into_dict(normal_loss_angle_diff_array: np.ndarray) -> tuple[dict, int]:
    n_rows = normal_loss_angle_diff_array.shape[0]
    data_dict = {
        "iteration": normal_loss_angle_diff_array[:, 0],
        "loss": normal_loss_angle_diff_array[:, 1],
        "normal_x": normal_loss_angle_diff_array[:, 2],
        "normal_y": normal_loss_angle_diff_array[:, 3],
        "normal_z": normal_loss_angle_diff_array[:, 4],
        "angle_diff": normal_loss_angle_diff_array[:, 5],
    }
    return data_dict, n_rows


def run_known_symmetry_test(mesh,
                            center,
                            normals,
                            angles,
                            args):
    print("Mesh vertices: {}".format(mesh.points().shape[0]))
    print("Original normal: {}".format(normals[0, :]))

    used_args = {
        "desviacion": angles[1:]
    }
    for key in shift_params_keys:
        if key == "delta_radius":
            used_args[key] = []
        else:
            used_args[key] = getattr(args, key)
    origin_circle = Circle(center, 1, normals[0, :])
    previous_loss, _ = compute_symmetry_chamfer_distance_from_mesh(mesh,
                                                                   origin_circle,
                                                                   args.n_sample_points,
                                                                   args.n_angles)

    resulting_normals = []
    for i in range(1, len(angles)):
        # TODO: delete this!
        if angles[i] == 15:
            used_args["desviacion"] = [15]

            print("Shifted normal: {} ({} degrees)".format(normals[i, :], angles[i]))
            rad = angles[i] * np.pi / 180
            if args.test_type == "sanity":
                adjusted_delta_radius = np.tan(rad)
            else:
                adjusted_delta_radius = args.delta_radius
            used_args["delta_radius"].append(adjusted_delta_radius)
            sym_circle = Circle(center, 1, normals[i, :])
            best_normals, new_generator_circle, num_its, _ = iterative_symmetry_shift(mesh,
                                                                                      sym_circle,
                                                                                      adjusted_delta_radius,
                                                                                      args.decrease_factor,
                                                                                      args.phi_simmetries,
                                                                                      args.epsilon_radius,
                                                                                      args.chi_convergence,
                                                                                      args.n_sample_points,
                                                                                      args.n_angles,
                                                                                      args.show_intermediate_axes)
            num_its = num_its[-1]
            best_loss_normal = np.copy(best_normals[-1, :])
            resulting_normals.append(np.hstack([num_its, best_loss_normal]))
            if args.show_ps_results:
                ps_test = show_mesh_with_all_found_axes(mesh, new_generator_circle, best_normals[:, 1:])
                ps_test.register_curve_network(f"Original Symmetry", np.array(
                    [-origin_circle.n + origin_circle.c, origin_circle.n + origin_circle.c]), np.array([[0, 1]]),
                                               radius=0.002)
                ps_test.show()
            print("Angle diff with original: {}".format(myangle(new_generator_circle.n, origin_circle.n)))
    resulting_normals = np.stack(resulting_normals)
    normal_angle_diff = compute_normal_angles_change(resulting_normals, origin_circle)

    if args.plt_y_axis == "loss":
        col_offset = 1
    else:
        col_offset = 0
    col_slice = [col_offset] + list(range(2, normal_angle_diff.shape[1]))

    print(normal_angle_diff)
    # plot_loss_and_angle_change(normal_angle_diff[:, col_slice], args.plt_y_axis)

    data, n_rows = convert_test_array_into_dict(normal_angle_diff)
    save_normal_angle_diff_into_csv(args.file, n_rows, data, **used_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test iterative shift over meshes that have known symmetries')
    add_shift_args(parser)
    parser.add_argument("--axis_file",
                        default='files/sym_test/mesh2.txt',
                        type=str,
                        help='File with axis information')
    parser.add_argument("--file",
                        default='files/sym_test/mesh2_man_sim.off',
                        type=str,
                        help='File to use')
    parser.add_argument("--test_type",
                        default='normal',
                        choices=["normal", "sanity"],
                        type=str,
                        help='File to use')
    parser.add_argument('--show_ps_results',
                        default=False,
                        action='store_true',
                        help="True if you want to display results.")
    parser.add_argument("--plt_y_axis",
                        default='loss',
                        choices=["loss", "iteration"],
                        type=str,
                        help='File to use')

    args = parser.parse_args()

    center, normals, angles = parse_axis_perturbation_file(args.axis_file)

    mesh = openmesh.read_trimesh(args.file)

    run_known_symmetry_test(mesh, center, normals, angles, args)
