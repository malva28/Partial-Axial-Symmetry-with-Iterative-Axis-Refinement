import argparse
import os
import numpy as np
from sklearn import neighbors

import polyscope as ps
import trimesh
import laplace
import openmesh
from signature import compute_signature
from fps import compute_fps
from supporting_circles import compute_supporting_circles


def generate_circle_node_edges(circle, n_nodes=10):
    c, r, n = circle

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
        edges.append([i, (i+1)%n_nodes])

    return np.array(nodes), np.array(edges)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mesh signature visualization')
    parser.add_argument('--n_basis', default='100', type=int, help='Number of basis used')
    parser.add_argument('--approx', default='cotangens', choices=laplace.approx_methods(), type=str,
                        help='Laplace approximation to use')
    parser.add_argument('--file', default='cat0.off', type=str, help='File to use')
    args = parser.parse_args()

    print("Reading Mesh")
    mesh = openmesh.read_trimesh(args.file)

    # Signature extraction
    print("Computing Signatures")
    signature_extractor = compute_signature(args.file, args)
    hks = signature_extractor.heat_signatures(10)

    # FPS
    print("FPSampling")
    point_cloud = mesh.points()
    sample_points, sample_indices = compute_fps(args.file,
                                                min(point_cloud.shape[0], max(point_cloud.shape[0]//100, 100)),
                                                pc=point_cloud)

    # knn
    print("(KNN) Finding Similar HKS points to the sampled points")
    nbrs = neighbors.NearestNeighbors(n_neighbors=100, algorithm='ball_tree').fit(hks)
    nbrs_distances, nbrs_indices = nbrs.kneighbors(hks[sample_indices])
    print("Computing Supporting Circles")
    compute_supporting_circles(point_cloud[nbrs_indices])

    ps.init()

    ps_mesh = ps.register_surface_mesh("mesh", mesh.points(), mesh.face_vertex_indices())
    ps_mesh.add_scalar_quantity("HKS (02nd descriptor)", hks[:, 1], cmap='coolwarm')
    ps_cloud = ps.register_point_cloud("sample points", sample_points)
    ps_similar = ps.register_point_cloud("similar hks points", point_cloud[nbrs_indices[30]])

    ps.show()
