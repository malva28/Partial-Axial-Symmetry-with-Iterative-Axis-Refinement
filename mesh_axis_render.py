import polyscope as ps
import numpy as np
from symmetric_support_estimator import sort_points_in_z_axis, compute_symmetry_count_scalar_quantity
from transformations import normalize, reorient_point_cloud, reorient_point_cloud_by_angles
import os


def generate_circle_node_edges(circle: "Circle", n_nodes=10):
    c, r, n = circle.get_c_r_n_tuple()

    # let v1, v2, n an orthonormal system
    v1 = np.array([n[1], -n[0], 0])
    if np.array_equal(v1, np.zeros(3)):  # n is a Z(+|-) vector, so v1 has to be calculated in another way
        v1 = np.array([1, 0, 0])  # but any (X|Y)(+|-) will do it.

    v2 = np.cross(n, v1)

    # make them orthonormal
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    nodes = []
    edges = []
    for i in range(0, n_nodes):
        theta = i * 2 * np.pi / n_nodes
        nodes.append(c + r * (v1 * np.cos(theta) + v2 * np.sin(theta)))
        edges.append([i, (i + 1) % n_nodes])

    return np.array(nodes), np.array(edges)


def show_mesh_with_partial_axis(mesh, generator_circle, symmetric_support_threshold, phi, theta):

    point_cloud = mesh.points()
    normalize(point_cloud)
    # Reorientation
    reorient_point_cloud_by_angles(point_cloud, phi, theta)
    # Symmetric Support
    max_dist = 1  # As the object is normalized to 1, we can use that
    print("Computing symmetric suppport")
    sorted_point_cloud, sorted_fvi = sort_points_in_z_axis(point_cloud, mesh.face_vertex_indices())
    symmetry_levels = compute_symmetry_count_scalar_quantity(sorted_point_cloud,
                                                             symmetric_support_threshold * max_dist)

    ps.set_up_dir("z_up")
    ps.init()

    # ps_mesh = ps.register_surface_mesh("mesh", mesh.points(), mesh.face_vertex_indices())
    # ps_mesh.add_scalar_quantity("HKS (02nd descriptor)", hks[:, 1], cmap='coolwarm')
    ps_mesh = ps.register_surface_mesh("sorted_mesh", sorted_point_cloud, sorted_fvi)
    ps_mesh.add_scalar_quantity("Symmetry Levels", symmetry_levels, cmap='coolwarm')

    # Generator Circle & Axis
    circle_nodes, circle_edges = generate_circle_node_edges(generator_circle)
    ps.register_curve_network(f"Generator Circle", circle_nodes, circle_edges, radius=0.005)
    ps_generator_center = ps.register_point_cloud("Generator Center", np.array([generator_circle.c]), radius=0.01)
    ps_generator_center.add_vector_quantity("Normal", np.array([generator_circle.n]))
    ps.register_curve_network(f"Generator Axis", np.array(
        [-generator_circle.n + generator_circle.c, generator_circle.n + generator_circle.c]), np.array([[0, 1]]),
                              radius=0.002)
    ps.show()


def take_screenshots(file_name, approx, n_basis):
    ps.set_screenshot_extension(".jpg")

    name = os.path.splitext(file_name)[0] + '-' + approx + '-' + str(n_basis)
    path = os.path.join('results', name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ps.screenshot(path + str("_z_up.jpg"))
    ps.set_up_dir("x_up")
    ps.screenshot(path + str("_x_up.jpg"))
    ps.set_up_dir("y_up")
    ps.screenshot(path + str("_y_up.jpg"))
