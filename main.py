import argparse
import os
from sys import exit
import numpy as np
from sklearn import neighbors

import polyscope as ps
import trimesh
import laplace
import openmesh

import signature
from signature import compute_signature
from fps import compute_fps
from supporting_circles import compute_supporting_circles, Circle
from generator_axis import compute_generator_axis
from transformations import normalize, reorient_point_cloud, reorient_circle
from symmetric_support_estimator import sort_points_in_z_axis, compute_symmetry_count_scalar_quantity


def generate_circle_node_edges(circle: Circle, n_nodes=10):
    c, r, n = circle.get_c_r_n_tuple()

    # let v1, v2, n an orthonormal system
    v1 = np.array([n[1], -n[0], 0])
    if np.array_equal(v1, np.zeros(3)):  # n is a Z(+|-) vector, so v1 has to be calculated in another way
        v1 = np.array([1, 0, 0])  # but any (X|Y)(+|-) will do it.

    v2 = np.cross(n, v1)

    # make them orthonormal
    v1 = v1/np.linalg.norm(v1)
    v2 = v2/np.linalg.norm(v2)

    nodes = []
    edges = []
    for i in range(0, n_nodes):
        theta = i * 2 * np.pi/n_nodes
        nodes.append(c + r * (v1 * np.cos(theta) + v2 * np.sin(theta)))
        edges.append([i, (i+1) % n_nodes])

    return np.array(nodes), np.array(edges)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mesh signature visualization')
    parser.add_argument('--file', default='files/cat0.off', type=str, help='File to use')
    # Laplace-related
    parser.add_argument('--n_basis', default='100', type=int, help='Number of basis used')
    parser.add_argument('--approx', default='cotangens', choices=laplace.approx_methods(), type=str,
                        help='Laplace approximation to use')
    parser.add_argument('--signature', default='heat', choices=signature.kernel_signatures(), type=str,
                        help='Kernel signature to use')
    # FPS-related
    parser.add_argument('--n_samples', default=30, type=int, help='number of points to sample with FPS')
    # Supporting Circles-related
    parser.add_argument('--n_candidates_per_circle', default=500, type=int,
                        help='Number of circle candidates from where to choose each single Supporting Circle.')
    parser.add_argument('--circle_candidate_threshold', default=0.005, type=float,
                        help='Threshold to consider a point part of a circle candidate. Distance to circle candidate '
                             '(float representing percentage of diagonal)')
    # Clustering-related (Generator Axis)
    parser.add_argument('--angular_r', default=0.015, type=float,
                        help='Maximum distance point-centroid in the angular-clustering.')
    parser.add_argument('--angular_s', default=0.03, type=float,
                        help='Minimum distance to a distant point in the angular-clustering.')
    parser.add_argument('--angular_k', default=10, type=int,
                        help='Minimum number of points per cluster in the angular-clustering.')
    parser.add_argument('--axis_r', default=0.25, type=float,
                        help='Maximum distance point-centroid in the axis-clustering.')
    parser.add_argument('--axis_s', default=0.5, type=float,
                        help='Minimum distance to a distant point in the axis-clustering.')
    parser.add_argument('--axis_k', default=5, type=int,
                        help='Minimum number of points per cluster in the axis-clustering.')
    # Symmetric Support-related
    parser.add_argument('--symmetric_support_threshold', default=0.01, type=float,
                        help='Threshold to consider a point affected by axial symmetry. Distance to the axial circle. '
                             '(float representing percentage of diagonal)')
    # Visual related
    parser.add_argument('--visual', default=True, action='store_true', help="True if you want to display results.")
    parser.add_argument('--no-visual', dest='visual', action='store_false')

    args = parser.parse_args()

    try:
        np.random.seed(123)

        print("Reading Mesh")
        mesh = openmesh.read_trimesh(args.file)
        point_cloud = mesh.points()
        normalize(point_cloud)

        # Signature extraction
        print("Computing Signatures")
        signature_extractor = compute_signature(args.file, args)
        hks = signature_extractor.signatures(args.signature, 300)

        print("Is nan?:", np.isnan(np.sum(hks)))

        # FPS
        n_samples = max(args.n_samples, point_cloud.shape[0]//200)  # Use at least 0.5% of the points
        n_samples = min(n_samples, point_cloud.shape[0])  # But no more than 100% of the points
        print(f"FPSampling with {n_samples}")
        sample_points, sample_indices = compute_fps(args.file,
                                                    n_samples,
                                                    pc=point_cloud)

        # knn
        print(f"(KNN) Finding Similar {args.signature} signature points to the sampled points")
        nbrs = neighbors.NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(hks)
        nbrs_distances, nbrs_indices = nbrs.kneighbors(hks[sample_indices])

        # Supporting Circles
        print("Computing Supporting Circles")
        max_dist = 1  # As the object is normalized to 1, we can use that
        s_circles, s_circles_votes = compute_supporting_circles(point_cloud[nbrs_indices],
                                                                args.n_candidates_per_circle,
                                                                args.circle_candidate_threshold*max_dist)

        # Generator axis
        print("Computing Generator Axis")
        best_s_circles, generator_circle = compute_generator_axis(s_circles,
                                                                  args.angular_r, args.angular_s, args.angular_k,
                                                                  args.axis_r, args.axis_s, args.axis_k)

        # Reorientation
        reorient_point_cloud(point_cloud, generator_circle)
        for s_circle in best_s_circles:
            reorient_circle(s_circle, generator_circle)
        reorient_circle(generator_circle, generator_circle)

        # Symmetric Support
        print("Computing symmetric suppport")
        sorted_point_cloud, sorted_fvi = sort_points_in_z_axis(point_cloud, mesh.face_vertex_indices())
        symmetry_levels = compute_symmetry_count_scalar_quantity(sorted_point_cloud,
                                                                 args.symmetric_support_threshold*max_dist)

        with open("log.txt", "a") as logf:
            logf.write(args.file + ", " + "0" + ", \n")
    except ValueError as e:
        print(e)
        with open("log.txt", "a") as logf:
            logf.write(args.file + ", " + "1" + ", " + e.__class__.__name__ + "\n")

        # clear .npz
        name = os.path.splitext(args.file)[0]
        path = os.path.join('data', name + '.npz')
        if os.path.exists(path):
            os.remove(path)
        path = os.path.join('data', str(name) + "fp_sampled" +
                            str(min(point_cloud.shape[0], max(point_cloud.shape[0] // 200, 30))) + '.npz')
        if os.path.exists(path):
            os.remove(path)

        exit(0)

    except Exception as e:
        print(e)
        with open("log.txt", "a") as logf:
            logf.write(args.file + ", " + "1" + ", " + e.__class__.__name__ + "\n")
        # ps.init()
        # ps_mesh = ps.register_surface_mesh("mesh", mesh.points(), mesh.face_vertex_indices())
        # print(hks.shape)
        # ps_mesh.add_scalar_quantity("HKS (02nd descriptor)", hks[:, 1], cmap='coolwarm')
        # ps.show()

        exit(0)

    if not args.visual:
        print("no visual")
        exit(0)

    ps.set_up_dir("z_up")
    ps.init()

    # ps_mesh = ps.register_surface_mesh("mesh", mesh.points(), mesh.face_vertex_indices())
    # ps_mesh.add_scalar_quantity("HKS (02nd descriptor)", hks[:, 1], cmap='coolwarm')
    ps_mesh = ps.register_surface_mesh("sorted_mesh", sorted_point_cloud, sorted_fvi)
    ps_mesh.add_scalar_quantity("Symmetry Levels", symmetry_levels, cmap='coolwarm')

    ps_cloud = ps.register_point_cloud("sample points", sample_points)
    # ps_similar = ps.register_point_cloud("similar hks points", point_cloud[nbrs_indices[10]])

    # All supporting circles
    """
    circle_centers = np.array([s_circle.c for s_circle in s_circles])
    ps_circle_centers = ps.register_point_cloud("Supporting Circle Centers", circle_centers)

    for i in range(len(s_circles)):
        circle_nodes, circle_edges = generate_circle_node_edges(s_circles[i])
        ps.register_curve_network(f"Supporting Circle {i+1:03d}, votes:{s_circles_votes[i]}", circle_nodes, circle_edges, radius=0.001)
        ps_circle_centers.add_vector_quantity("Normal", np.array([s_circle.n for s_circle in s_circles]))
    """

    # Best Circles Cluster
    """
    ps_best_circle_centers = ps.register_point_cloud(
        "Best Circle Centers",
        np.array([s_circle.c for s_circle in best_s_circles]))
    for i in range(len(best_s_circles)):
        circle_nodes, circle_edges = generate_circle_node_edges(best_s_circles[i])
        ps.register_curve_network(f"Best circle {i+1:03d}", circle_nodes, circle_edges, radius=0.003)
        ps_best_circle_centers.add_vector_quantity("Normal", np.array([s_circle.n for s_circle in best_s_circles]))
    """

    # Generator Circle & Axis
    circle_nodes, circle_edges = generate_circle_node_edges(generator_circle)
    ps.register_curve_network(f"Generator Circle", circle_nodes, circle_edges, radius=0.005)
    ps_generator_center = ps.register_point_cloud("Generator Center", np.array([generator_circle.c]), radius=0.01)
    ps_generator_center.add_vector_quantity("Normal", np.array([generator_circle.n]))
    ps.register_curve_network(f"Generator Axis", np.array(
        [-generator_circle.n + generator_circle.c, generator_circle.n + generator_circle.c]), np.array([[0, 1]]),
                              radius=0.002)

    ps.show()
