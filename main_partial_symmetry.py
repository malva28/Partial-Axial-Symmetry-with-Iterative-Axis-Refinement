import argparse
import os
import sys
from sys import exit, stderr
import traceback
import numpy as np
from sklearn import neighbors
import csv

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
import utils
from common_args import create_common_args, symmetry_param_keys
from mesh_axis_render import show_mesh_with_partial_axis, take_screenshots


class LogEntry:
    def __init__(self, project_path = ""):
        self.file = ""
        self.exit_status = 0
        self.additional = {}
        self.exception = None
        self.exc_tb = None
        self.project_path = project_path

    def get_deepest_frame_in_project(self):
        tb = self.exc_tb
        while tb.tb_next is not None:
            next_exc_filename = tb.tb_next.tb_frame.f_code.co_filename
            if self.project_path not in next_exc_filename:
                break
            tb = tb.tb_next
        return os.path.split(tb.tb_frame.f_code.co_filename)[1]

    def __str__(self):
        add_vals = [str(self.additional[k]) for k in self.additional]
        clean_exc_msg = str(self.exception).replace("\n", "")
        exc_name = self.exception.__class__.__name__ if self.exception else ""
        exc_filename = self.get_deepest_frame_in_project() if self.exc_tb else ""
        lineno = str(self.exc_tb.tb_lineno) if self.exc_tb else ""

        the_str = ", ".join([self.file,
                            str(self.exit_status),
                            *add_vals,
                            exc_name,
                            exc_filename,
                            lineno,
                            clean_exc_msg])
        return the_str


class Log:
    def __init__(self, log_name, project_path):
        self.log_name = log_name
        self.project_path = project_path

    def write_entry(self, log_entry):
        with open(self.log_name, "a") as logf:
            logf.write(str(log_entry) + "\n")

    def handle_exception_and_write_entry(self, log_entry, exception):
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(''.join(traceback.format_tb(exception.__traceback__)), file=stderr)
        print(exception, file=stderr)
        log_entry.exc_tb = exc_tb
        log_entry.exception = exception
        log_entry.exit_status = 1
        self.write_entry(log_entry)


if __name__ == '__main__':
    parser = create_common_args(description='Mesh signature visualization')
    args = parser.parse_args()

    try:
        project_dir = os.path.dirname(__file__)
        log_obj = Log("log.txt", project_dir)
        log_entry = LogEntry(project_path= project_dir)

        log_entry.file = args.file

        np.random.seed(123)

        print("Reading Mesh")
        mesh = openmesh.read_trimesh(args.file)
        point_cloud = mesh.points()
        normalize(point_cloud)

        # Signature extraction
        print("Computing Signatures")
        signature_extractor = compute_signature(args.file, args)
        hks = signature_extractor.signatures(args.signature, 300, log_entry=log_entry)

        print("Is nan?:", np.isnan(np.sum(hks)))

        log_entry.additional["nan"] = np.isnan(np.sum(hks))

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
        original_phi = -generator_circle.get_phi()
        original_theta = -generator_circle.get_theta()

        # Reorientation
        # reorient_point_cloud(point_cloud, generator_circle)
        for s_circle in best_s_circles:
            reorient_circle(s_circle, generator_circle)
        reorient_circle(generator_circle, generator_circle)

        # Symmetric Support
        print("Computing symmetric suppport")
        sorted_point_cloud, sorted_fvi = sort_points_in_z_axis(point_cloud, mesh.face_vertex_indices())
        symmetry_levels = compute_symmetry_count_scalar_quantity(sorted_point_cloud,
                                                                 args.symmetric_support_threshold*max_dist)

        # write log
        log_obj.write_entry(log_entry)

        # cache found circle
        args_dict = vars(args)
        utils.write_cache(args_dict, generator_circle, original_phi, original_theta)

    except ValueError as e:
        log_obj.handle_exception_and_write_entry(log_entry, e)

        # clear .npz
        name = os.path.splitext(args.file)[0]+'-'+args.approx+'-'+str(args.n_basis)
        path = os.path.join('data', name + '.npz')
        if os.path.exists(path):
            os.remove(path)
        path = os.path.join('data', str(name) + "fp_sampled" +
                            str(min(point_cloud.shape[0], max(point_cloud.shape[0] // 200, 30))) + '.npz')
        if os.path.exists(path):
            os.remove(path)

        exit(1)

    except Exception as e:
        log_obj.handle_exception_and_write_entry(log_entry, e)
        # ps.init()
        # ps_mesh = ps.register_surface_mesh("mesh", mesh.points(), mesh.face_vertex_indices())
        # print(hks.shape)
        # ps_mesh.add_scalar_quantity("HKS (02nd descriptor)", hks[:, 1], cmap='coolwarm')
        # ps.show()

        exit(1)

    if not args.visual:
        print("no visual")
        exit(0)

    ps = show_mesh_with_partial_axis(mesh, generator_circle, args.symmetric_support_threshold, original_phi, original_theta)
    ps.show()
    take_screenshots(args.file, args.approx, args.n_basis)
