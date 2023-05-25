import pandas as pd
import os
import sys
import subprocess
import functools
import numpy as np
import openmesh
import torch
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

from common_args import create_common_args, symmetry_param_keys
from supporting_circles import compute_supporting_circles, Circle
from mesh_axis_render import show_mesh_with_partial_axis, take_screenshots
from utils import cache_exists, cache_empty, read_cache, find_row_in_cache, get_generator_circle_from_cache
from compute_chamfer import compute_symmetry_chamfer_distance


def get_row_and_angles_from_mesh(mesh):
    df = read_cache()

    row, _ = find_row_in_cache(df, **parser_vals)

    phi_adjust = row.iloc[0]["phi_adjust"]
    theta_adjust = row.iloc[0]["theta_adjust"]
    return phi_adjust, theta_adjust, row


def use_from_cache(mesh, parser_vals):
    print("Using cached values for symmetry axis")
    phi_adjust, theta_adjust, row = get_row_and_angles_from_mesh(mesh)
    generator_circle = get_generator_circle_from_cache(row)
    print(generator_circle)

    show_mesh_with_partial_axis(
        mesh,
        generator_circle,
        args.symmetric_support_threshold,
        phi_adjust,
        theta_adjust,
        symmetric_support= False)
    return generator_circle



def run_initial_partial_symmetry():
    venv_exe = sys.executable
    flat_args = []
    for key in symmetry_param_keys:
        flat_args.append("--" + key)
        flat_args.append(str(arg_dict[key]))
    process = subprocess.run([venv_exe, 'main_partial_symmetry.py', *flat_args], check=True)
    ret = process.returncode
    print("Finished executing main_partial_symmetry with return code {}".format(ret))
    return ret


if __name__ == "__main__":
    parser = create_common_args(description='Mesh signature visualization')
    parser.add_argument('--cache', default=True, action='store_true', help="True if you want to retrieve initial axis "
                                                                           "from cache file.\nFalse if you want to "
                                                                           "reforce partial axis calculation")
    parser.add_argument('--no-cache', dest='cache', action='store_false')

    delta_radius = 0.1
    decrease_factor = 0.9
    phi_simmetries = 5
    min_iterations = 10
    epsilon_convergence = 1e-5

    args = parser.parse_args()
    arg_dict = vars(args)
    parser_vals = {key: arg_dict[key] for key in symmetry_param_keys}
    print("Reading Mesh")
    mesh = openmesh.read_trimesh(args.file)

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
        generator_circle = use_from_cache(mesh, parser_vals)
    point_cloud_tensor = torch.tensor(mesh.points(), dtype=torch.float32)
    point_cloud_tensor = point_cloud_tensor.unsqueeze(0)
    # previous_loss = compute_symmetry_chamfer_distance(mesh, generator_circle)
    previous_loss = compute_symmetry_chamfer_distance(point_cloud_tensor, generator_circle)
    print("Initial loss: {}".format(previous_loss))
    old_c, old_r, old_n = generator_circle.get_c_r_n_tuple()
    convergence = np.infty
    iterations = 0
    losses = np.zeros((phi_simmetries, 4))

    while iterations < min_iterations or convergence > epsilon_convergence:
        new_c = old_c + old_n
        variaton_circle = Circle(new_c, delta_radius, old_n)
        new_normal_endpoints, _ = variaton_circle.generate_circle_node_edges(
            phi_simmetries,
            np.random.uniform(0, np.pi, 1))
        for i in range(phi_simmetries):
            normal_endpoint = new_normal_endpoints[i]
            candidate_normal = normal_endpoint - old_c
            candidate_normal = candidate_normal/ np.linalg.norm(candidate_normal)
            candidate_circle = Circle(old_c, old_r, candidate_normal)
            # losses[i, 0] = compute_symmetry_chamfer_distance(mesh, candidate_circle)
            losses[i, 0] = compute_symmetry_chamfer_distance(point_cloud_tensor, candidate_circle)
            losses[i, 1:4] = candidate_normal
        i_min_found_loss = np.argmin(losses[:,0])
        if losses[i_min_found_loss, 0] < previous_loss:
            convergence = previous_loss - losses[i_min_found_loss, 0]
            old_n = losses[i_min_found_loss, 1:4]
            previous_loss = losses[i_min_found_loss, 0]
            print("New loss: {}, New normal: {}, convergence {}".format(previous_loss, old_n, convergence))
        else:
            delta_radius *= decrease_factor
            print("Decreased radius factor: {}. Iteration: {}".format(delta_radius, iterations))
        iterations += 1

    generator_circle = Circle(old_c, old_r, old_n)
    print("Converged after {} iterations.\nNew circle: {}".format(iterations, generator_circle))
    # phi, theta, _ = get_row_and_angles_from_mesh(mesh)

    show_mesh_with_partial_axis(
        mesh,
        generator_circle,
        args.symmetric_support_threshold,
        phi=0.0,
        theta=0.0,
        symmetric_support=False)






