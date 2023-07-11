
from transformations import calculate_angle_diffs_with_vector
from compute_chamfer import compute_symmetry_chamfer_distance, compute_symmetry_chamfer_distance_from_mesh
from supporting_circles import Circle
from progress_bar import print_progress_bar

import numpy as np
import polyscope as ps
import matplotlib.pyplot as plt
import pandas as pd
import os


def show_intermediate_normal(mesh,
                             initial_sym_circle: Circle,
                             var_circle_endpoints,
                             var_circle_edges,
                             num_normals,
                             candidate_normals,
                             new_normal):
    circle_nodes, circle_edges = initial_sym_circle.generate_circle_node_edges()
    ps.init()
    point_cloud = mesh.points()
    ps_mesh = ps.register_surface_mesh("sorted_mesh", point_cloud, mesh.face_vertex_indices())
    ps.register_curve_network(f"Generator Circle", circle_nodes, circle_edges, radius=0.005)
    ps.register_curve_network(f"Generator Circle Var", var_circle_endpoints, var_circle_edges, radius=0.005)
    ps_generator_center = ps.register_point_cloud("Generator Center", np.tile(np.array([initial_sym_circle.c]),
                                                                              [num_normals + 1, 1]),
                                                  radius=0.01)
    ps_generator_center.add_vector_quantity("candidates", np.vstack((initial_sym_circle.n, candidate_normals)))
    ps.register_curve_network(f"Generator Axis", np.array(
        [-initial_sym_circle.n + initial_sym_circle.c, initial_sym_circle.n + initial_sym_circle.c]),
                              np.array([[0, 1]]),
                              radius=0.002)
    ps.register_curve_network(f"New Axis", np.array(
        [-new_normal + initial_sym_circle.c, new_normal + initial_sym_circle.c]), np.array([[0, 1]]),
                              radius=0.002)

    ps.show()


def iterative_symmetry_shift(mesh,
                             sym_circle: Circle,
                             d_radius,
                             dec_factor,
                             p_symmetries,
                             e_radius,
                             c_convergence,
                             n_s_points,
                             num_angles,
                             show_intermediate_axes=False) -> tuple[np.ndarray, Circle, list[int], dict]:
    point_cloud = mesh.points()

    previous_loss, point_cloud_tensor = compute_symmetry_chamfer_distance_from_mesh(mesh,
                                                                                    sym_circle,
                                                                                    n_s_points,
                                                                                    num_angles)

    print("Mesh vertices: {}. Sample size: {}".format(point_cloud.shape[0], n_s_points))
    print("Radius of circle: {}.\n"
          "Angle between new normals and old normal (degrees): {}".format(d_radius,
                                                                          np.arctan(d_radius)*180/np.pi))
    print("Initial loss: {}".format(previous_loss))
    old_c, old_r, old_n = sym_circle.get_c_r_n_tuple()
    convergence = np.infty
    iterations = 0
    losses = np.zeros((p_symmetries, 4))
    num_not_found = 1

    zero_normal = np.zeros(len(old_n) + 1)
    zero_normal[0] = previous_loss
    zero_normal[1:4] = np.copy(old_n)

    best_normals = [zero_normal]
    additional = {
        "delta_radius": [d_radius],
        "decrease_factor": [dec_factor]
    }

    iteration_list = [iterations]

    # while iterations < min_iterations or delta_radius > epsilon_radius:
    while d_radius > e_radius and convergence > c_convergence ** 2:
        new_c = old_c + old_n
        variaton_circle = Circle(new_c, d_radius, old_n)
        new_normal_endpoints, endpoint_edges = variaton_circle.generate_random_circle_node_edges(
            p_symmetries)
        candidates_normals = np.copy(new_normal_endpoints)
        candidates_normals -= old_c
        norms = np.tile(1 / np.linalg.norm(candidates_normals[..., None], axis=1), [1, 3])
        candidates_normals *= norms
        # debug
        if show_intermediate_axes:
            show_intermediate_normal(mesh,
                                     sym_circle,
                                     new_normal_endpoints,
                                     endpoint_edges,
                                     p_symmetries,
                                     candidates_normals,
                                     old_n)

        for i in range(p_symmetries):
            print_progress_bar(i, p_symmetries, print_end="")
            candidate_normal = candidates_normals[i]
            # normal_endpoint = new_normal_endpoints[i]
            # candidate_normal = normal_endpoint - old_c
            # candidate_normal = candidate_normal/ np.linalg.norm(candidate_normal)
            candidate_circle = Circle(old_c, old_r, candidate_normal)
            # losses[i, 0] = compute_symmetry_chamfer_distance(mesh, candidate_circle)
            losses[i, 0] = compute_symmetry_chamfer_distance(point_cloud_tensor, candidate_circle, num_angles)
            losses[i, 1:4] = candidate_normal
        i_min_found_loss = np.argmin(losses[:, 0])
        print()
        print("min_loss: {}, convergence: {}".format(losses[i_min_found_loss, 0],
                                                     losses[i_min_found_loss, 0] - previous_loss))
        iterations += 1
        if losses[i_min_found_loss, 0] < previous_loss:
            old_n = np.copy(losses[i_min_found_loss, 1:4])
            convergence = np.abs(losses[i_min_found_loss, 0] - previous_loss)
            previous_loss = losses[i_min_found_loss, 0]
            best_normals.append(np.copy(losses[i_min_found_loss]))
            additional["delta_radius"].append(d_radius)
            iteration_list.append(iterations)
            additional["decrease_factor"].append(dec_factor ** num_not_found)
            if convergence > c_convergence:
                num_not_found = 1
            print("New normal: {}".format(old_n))
        else:
            d_radius *= (dec_factor ** num_not_found)
            num_not_found += 1
            print("Decreased radius factor: {}. Iteration: {}".format(d_radius, iterations))


    generator_circle = Circle(old_c, old_r, old_n)
    print("Converged after {} iterations.\nNew circle: {}".format(iterations, generator_circle))
    # phi, theta, _ = get_row_and_angles_from_mesh(mesh)
    best_normals = np.stack(best_normals)
    return best_normals, generator_circle, iteration_list, additional


def compute_normal_angles_change(normal_array, initial_sym_circle: Circle):
    prev_row_len = len(normal_array[0])

    best_normals_with_angle_diff = np.zeros((len(normal_array), prev_row_len + 1))
    best_normals_with_angle_diff[:, 0:prev_row_len] = np.stack(normal_array)

    best_normals_with_angle_diff[0:, prev_row_len] = calculate_angle_diffs_with_vector(normal_array[0:, prev_row_len-3:prev_row_len],
                                                                                       initial_sym_circle.n)

    print(best_normals_with_angle_diff)
    return best_normals_with_angle_diff


def plot_loss_and_angle_change(normal_angle_diff_array, first_y_label="loss"):
    iteration_array = np.arange(normal_angle_diff_array.shape[0])
    len_row = normal_angle_diff_array.shape[1]
    # plot results
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('n iteraciones')
    ax1.set_ylabel(first_y_label, color=color)
    ax1.plot(iteration_array, normal_angle_diff_array[:, 0], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    # ax1.set_xscale('log')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('angles (degrees)', color=color)  # we already handled the x-label with ax1
    ax2.plot(iteration_array, normal_angle_diff_array[:, len_row-1:len_row] *180/np.pi,
             color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    # ax2.set_xscale('log')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def convert_nlad_array_into_dict(normal_loss_angle_diff_array: np.ndarray, iteration_array: list = []) -> tuple[dict, int]:
    if len(iteration_array) == 0:
        iteration_array = np.arange(normal_loss_angle_diff_array.shape[0])
    n_rows = normal_loss_angle_diff_array.shape[0]
    data_dict = {
        "iteration": iteration_array,
        "loss": normal_loss_angle_diff_array[:, 0],
        "normal_x": normal_loss_angle_diff_array[:, 1],
        "normal_y": normal_loss_angle_diff_array[:, 2],
        "normal_z": normal_loss_angle_diff_array[:, 3],
        "angle_diff": normal_loss_angle_diff_array[:, 4],
    }
    return data_dict, n_rows


def save_normal_angle_diff_into_csv(file_path, n_rows, data_dict, **params):
    # save to a pandas file
    time_now = pd.Timestamp.now().strftime('%Y-%m-%d_%X')
    data_dict["timestamp"] = [time_now] * n_rows

    for key in params:
        val = params[key]
        if not isinstance(val, list) and not isinstance(val, np.ndarray):
            val = [val]
        if len(val) == 1:
            val = np.tile(val, n_rows)
        data_dict[key] = val

    axis_df = pd.DataFrame(data_dict)
    axis_file_name = os.path.join("results", "axis", os.path.splitext(os.path.basename(file_path))[0] + ".csv")
    if os.path.exists(axis_file_name):
        axis_df.to_csv(axis_file_name, mode="a", index=False, header=False)
    else:
        axis_df.to_csv(axis_file_name, index=False)


def show_mesh_with_all_found_axes(
        mesh,
        gen_circle: Circle,
        normals,
        normal_labels=[],
        normal_radius=[]):
    n_normals = len(normals)
    point_cloud = mesh.points()
    ps.set_up_dir("z_up")
    ps.init()
    ps_mesh = ps.register_surface_mesh("sorted_mesh", point_cloud, mesh.face_vertex_indices())

    circle_nodes, circle_edges = gen_circle.generate_circle_node_edges()
    ps.register_curve_network(f"Generator Circle", circle_nodes, circle_edges, radius=0.005)
    ps_generator_center = ps.register_point_cloud("Generator Center", np.array([gen_circle.c]*n_normals),
                                                  radius=0.01)
    ps_generator_center.add_vector_quantity("Candidate normals", normals)
    ps.register_curve_network(f"Generator Axis", np.array(
        [-gen_circle.n + gen_circle.c, gen_circle.n + gen_circle.c]), np.array([[0, 1]]),
                              radius=0.002)
    cmap = plt.cm.get_cmap("hsv", n_normals+1)
    color_array = [cmap(i)[:-1] for i in range(n_normals)]
    color_array = np.array(color_array)
    if len(normal_labels) != n_normals:
        normal_labels = [i+1 for i in range(n_normals)]
    if len(normal_radius) != n_normals:
        normal_radius = [0.0005 for _ in range(n_normals)]
    for i in range(n_normals):
        normal = normals[i,:]
        color = color_array[i,:]
        label = normal_labels[i]
        radius = normal_radius[i]
        ps.register_curve_network("Axis {}".format(label), np.array(
            [-normal + gen_circle.c, normal + gen_circle.c]), np.array([[0, 1]]),
                                  radius=radius,
                                  color=color,
                                  transparency=0.8)
    return ps