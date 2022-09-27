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

    ps.init()

    ps_mesh = ps.register_surface_mesh("mesh", mesh.points(), mesh.face_vertex_indices())
    ps_mesh.add_scalar_quantity("HKS (02nd descriptor)", hks[:, 1], cmap='coolwarm')
    ps_cloud = ps.register_point_cloud("sample points", sample_points)

    ps.show()
