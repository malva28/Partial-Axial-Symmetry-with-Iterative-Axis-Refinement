import polyscope as ps
import numpy as np
from symmetric_support_estimator import sort_points_in_z_axis, compute_symmetry_count_scalar_quantity
from transformations import normalize, reorient_point_cloud, reorient_point_cloud_by_angles
import os





def show_mesh_with_partial_axis(mesh, generator_circle, symmetric_support_threshold, phi, theta, symmetric_support=True):

    point_cloud = mesh.points()
    normalize(point_cloud)
    # Reorientation
    reorient_point_cloud_by_angles(point_cloud, phi, theta)
    # Symmetric Support
    max_dist = 1  # As the object is normalized to 1, we can use that
    print("Computing symmetric suppport")

    sorted_point_cloud, sorted_fvi = sort_points_in_z_axis(point_cloud, mesh.face_vertex_indices())
    if symmetric_support:
        symmetry_levels = compute_symmetry_count_scalar_quantity(sorted_point_cloud,
                                                             symmetric_support_threshold * max_dist)

    ps.set_up_dir("z_up")
    ps.init()

    # ps_mesh = ps.register_surface_mesh("mesh", mesh.points(), mesh.face_vertex_indices())
    # ps_mesh.add_scalar_quantity("HKS (02nd descriptor)", hks[:, 1], cmap='coolwarm')
    ps_mesh = ps.register_surface_mesh("sorted_mesh", sorted_point_cloud, sorted_fvi)
    if symmetric_support:
        ps_mesh.add_scalar_quantity("Symmetry Levels", symmetry_levels, cmap='coolwarm')

    # Generator Circle & Axis
    circle_nodes, circle_edges = generator_circle.generate_circle_node_edges()
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
