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
from supporting_circles import compute_supporting_circles, Circle, get_phi_from_normal, get_theta_from_normal
from mesh_axis_render import show_mesh_with_partial_axis, take_screenshots
from utils import cache_exists, cache_empty, read_cache, find_row_in_cache, get_generator_circle_from_cache
from compute_chamfer import compute_symmetry_chamfer_distance
from progress_bar import print_progress_bar
import polyscope as ps
import matplotlib.pyplot as plt


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
    decrease_factor = 0.8
    phi_simmetries = 12
    #min_iterations = 10
    epsilon_radius = 1e-5
    chi_convergence = 1e-3
    n_sample_points = 4000
    n_angles = 12

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
    verts = torch.tensor(mesh.points(), dtype=torch.float32)
    faces = torch.tensor([[fi.idx() for fi in mesh.fv(face)] for face in mesh.faces()], dtype=torch.float32)
    trg_mesh = Meshes(verts=[verts], faces=[faces])
    # this method applies uniform sampling, but proportional to face area
    point_cloud_tensor = sample_points_from_meshes(trg_mesh, n_sample_points)

    #point_cloud_tensor = torch.tensor(mesh.points(), dtype=torch.float32)
    #point_cloud_tensor = point_cloud_tensor.unsqueeze(0)
    # previous_loss = compute_symmetry_chamfer_distance(mesh, generator_circle)
    previous_loss = compute_symmetry_chamfer_distance(point_cloud_tensor, generator_circle)
    origin_phi = generator_circle.get_phi()
    origin_theta = generator_circle.get_theta()

    print("Initial loss: {}".format(previous_loss))
    old_c, old_r, old_n = generator_circle.get_c_r_n_tuple()
    convergence = np.infty
    iterations = 0
    losses = np.zeros((phi_simmetries, 4))
    num_not_found = 1

    zero_normal = np.zeros(len(old_n)+1)
    zero_normal[0] = previous_loss
    zero_normal[1:4] = np.copy(old_n)

    best_normals = [zero_normal]

    circle_nodes, circle_edges = generator_circle.generate_circle_node_edges()
    show_intermediate_axes = False

    #while iterations < min_iterations or delta_radius > epsilon_radius:
    while delta_radius > epsilon_radius and convergence > chi_convergence**2:
        new_c = old_c + old_n
        variaton_circle = Circle(new_c, delta_radius, old_n)
        new_normal_endpoints, endpoint_edges = variaton_circle.generate_random_circle_node_edges(
            phi_simmetries)
        candidates_normals = np.copy(new_normal_endpoints)
        candidates_normals -= old_c
        norms = np.tile( 1/np.linalg.norm(candidates_normals[..., None], axis=1),[1,3])
        candidates_normals *= norms

        # debug
        if show_intermediate_axes:
            ps.init()
            ps_mesh = ps.register_surface_mesh("sorted_mesh", mesh.points(), mesh.face_vertex_indices())
            ps.register_curve_network(f"Generator Circle", circle_nodes, circle_edges, radius=0.005)
            ps.register_curve_network(f"Generator Circle Var", new_normal_endpoints, endpoint_edges, radius=0.005)
            ps_generator_center = ps.register_point_cloud("Generator Center", np.tile(np.array([generator_circle.c]), [phi_simmetries+1, 1]), radius=0.01)
            ps_generator_center.add_vector_quantity("candidates", np.vstack((generator_circle.n, candidates_normals)))
            ps.register_curve_network(f"Generator Axis", np.array(
                [-generator_circle.n + generator_circle.c, generator_circle.n + generator_circle.c]), np.array([[0, 1]]),
                                      radius=0.002)
            ps.register_curve_network(f"New Axis", np.array(
                [-old_n+ generator_circle.c, old_n + generator_circle.c]), np.array([[0, 1]]),
                                      radius=0.002)

            ps.show()

        for i in range(phi_simmetries):
            print_progress_bar(i, phi_simmetries, print_end="")
            candidate_normal = candidates_normals[i]
            #normal_endpoint = new_normal_endpoints[i]
            #candidate_normal = normal_endpoint - old_c
            #candidate_normal = candidate_normal/ np.linalg.norm(candidate_normal)
            candidate_circle = Circle(old_c, old_r, candidate_normal)
            # losses[i, 0] = compute_symmetry_chamfer_distance(mesh, candidate_circle)
            losses[i, 0] = compute_symmetry_chamfer_distance(point_cloud_tensor, candidate_circle, n_angles)
            losses[i, 1:4] = candidate_normal
        i_min_found_loss = np.argmin(losses[:,0])
        print()
        print("min_loss: {}, convergence: {}".format(losses[i_min_found_loss, 0], losses[i_min_found_loss, 0] -previous_loss))
        if losses[i_min_found_loss, 0] < previous_loss:
            old_n = np.copy(losses[i_min_found_loss, 1:4])
            convergence = np.abs(losses[i_min_found_loss, 0] - previous_loss)
            previous_loss = losses[i_min_found_loss, 0]
            best_normals.append(np.copy(losses[i_min_found_loss]))
            if convergence > chi_convergence:
                num_not_found = 1
            print("New normal: {}".format(old_n))
        else:
            delta_radius *= (decrease_factor ** num_not_found)
            num_not_found += 1
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

    if len(best_normals) > 0:
        prev_row_len = len(best_normals[0])

        best_normals_with_angle_diff = np.zeros((len(best_normals), prev_row_len+2))
        best_normals_with_angle_diff[:, 0:prev_row_len] = np.stack(best_normals)

        best_normals_with_angle_diff[1:, prev_row_len] = get_phi_from_normal(best_normals_with_angle_diff[1:,1:4]) -origin_phi
        best_normals_with_angle_diff[1:, prev_row_len+1] = get_theta_from_normal(best_normals_with_angle_diff[1:,1:4]) - origin_theta
        iteration_array = np.arange(best_normals_with_angle_diff.shape[0])

        print(best_normals_with_angle_diff)

        # plot results
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('n iteraciones')
        ax1.set_ylabel('loss', color=color)
        ax1.plot(iteration_array, best_normals_with_angle_diff[:,0], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        #ax1.set_xscale('log')

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('angles', color=color)  # we already handled the x-label with ax1
        ax2.plot(iteration_array, best_normals_with_angle_diff[:,prev_row_len:prev_row_len+2],
                 color=color,
                 label=["phi diff", "theta diff"])
        ax2.legend()
        ax2.tick_params(axis='y', labelcolor=color)
        #ax2.set_xscale('log')

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()








