import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import openmesh
from supporting_circles import Circle
from iterative_shift import show_mesh_with_all_found_axes
from symmetry_test import parse_axis_perturbation_file
from transformations import rad_to_degree, linear_range_map, reorient_point_cloud_by_angles, normalize

from main import get_row_and_angles_from_mesh
from utils import get_generator_circle_from_cache
import functools


def get_cmap(n, name='hsv'):
    """
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.
    reference: https://matplotlib.org/1.2.1/mpl_examples/pylab_examples/show_colormaps.pdf
    """
    return plt.colormaps[name].resampled(n)


def apply_filters(dataframe, filter_dict):
    filtered = dataframe

    for filter in filter_dict:
        filtered = filtered.loc[getattr(dataframe[filter["col"]], filter["op"])(filter["val"])]

    return filtered


def calc_mean_std(dataframe,
                  filter_dict,
                  grouping_col,
                  y_colnames):
    filtered_i = apply_filters(dataframe, filter_dict)
    grouped = filtered_i.groupby(grouping_col, as_index=False)
    # mean_i = grouped.mean(numeric_only=True)
    mean_i = grouped.mean(numeric_only=True)
    x = mean_i[grouping_col]
    mean_i = mean_i[y_colnames]
    std_i = grouped.std(numeric_only=True)[y_colnames]

    return x, mean_i, std_i


def add_desviacion(dataframe):
    sample_group = dataframe.groupby("timestamp")
    first_per_sample = sample_group.nth(0)

    n_rows = dataframe.shape[0]

    dataframe["desviacion"] = np.ones(n_rows)

    degrees = np.arctan(first_per_sample["delta_radius"])*180/np.pi
    timestamps = list(first_per_sample.index)

    for i in range(len(timestamps)):
        timestamp = timestamps[i]
        degree = degrees[i]
        n_affected_rows = dataframe.loc[dataframe["timestamp"] == timestamp, "desviacion"].shape[0]

        dataframe.loc[dataframe["timestamp"] == timestamp, "desviacion"] = np.ones(n_affected_rows)*degree
    return dataframe


def show_known_syms_ps():
    num_csvs = [1, 2, 4, 5, 6, 7]

    class TestType:
        phi_symmetries = "phi"
        n_angles = "n_angles"
        desviacion = "desviacion"

    test_type = TestType.desviacion

    if test_type == TestType.phi_symmetries:
        filters = [
            {
                "col": "chi_convergence",
                "op": "__eq__",
                "val": 0.0001
            },
            {
                "col": "n_angles",
                "op": "__eq__",
                "val": 6
            },
            {
                "col": "phi_simmetries",
                "op": "__gt__",
                "val": 2
            },
            {
                "col": "desviacion",
                "op": "__eq__",
                "val": 15
            }
        ]
        groups = ["phi_simmetries"]

    elif test_type == TestType.n_angles:
        filters = [
            {
                "col": "chi_convergence",
                "op": "__eq__",
                "val": 0.0001
            },
            {
                "col": "n_angles",
                "op": "__gt__",
                "val": 1
            },
            {
                "col": "phi_simmetries",
                "op": "__eq__",
                "val": 16
            },
            {
                "col": "desviacion",
                "op": "__eq__",
                "val": 15
            }
        ]
        groups = ["n_angles"]
    elif test_type == TestType.desviacion:
        filters = [
            {
                "col": "chi_convergence",
                "op": "__eq__",
                "val": 0.0001
            },
            {
                "col": "n_angles",
                "op": "__eq__",
                "val": 6
            },
            {
                "col": "phi_simmetries",
                "op": "__eq__",
                "val": 16
            }
        ]
        groups = ["desviacion"]

    for num_csv in num_csvs:
        file_name = "mesh{}_rand_man_sim.csv".format(num_csv)
        mesh_name = "mesh{}_man_sim.off".format(num_csv)
        center_name = "mesh{}.txt".format(num_csv)

        center, normals, _ = parse_axis_perturbation_file(os.path.join("files", "sym_test", center_name))
        mesh = openmesh.read_trimesh(os.path.join("files", "sym_test", mesh_name))

        pd_res = pd.read_csv(os.path.join("results", "axis", file_name))
        pd_res["angle_diff"] = rad_to_degree(pd_res["angle_diff"])

        pd_filtered = apply_filters(pd_res, filters)

        group_by = pd_filtered.groupby(groups)

        shown_y_columns = ["loss", "angle_diff"]
        mean_i = group_by.mean(numeric_only=True)[["normal_x", "normal_y", "normal_z"]]
        x = mean_i.index.get_level_values(0)
        std_i = group_by.std(numeric_only=True)

        std_ang = std_i["angle_diff"].to_numpy()

        ps = show_mesh_with_all_found_axes(mesh,
                                           Circle(center, 1, normals[0, :]),
                                           mean_i.to_numpy(),
                                           ["loss angs {}".format(i) for i in x],
                                           [0.0015 for i in x])
                                           #linear_range_map(std_ang, [std_ang.min(), std_ang.max()], [0.0001, 0.001]))

        ps.set_ground_plane_mode("shadow_only")
        ps.show()


def show_dataset_ps():
    dirs_file = ["results", "axis"]
    dirs_mesh = ["files", "Larco"]

    for f in os.listdir(os.path.join(*dirs_file)):
        if "decimated" in f and functools.reduce(lambda a,b: a or b, [n in f for n in ["08", "10", "12", "13", "18", "24", "25", "27", "28", "29", "39", "41"]]):
            num_csv = f[:2]
            file_name = "{}_decimated.csv".format(str(num_csv).zfill(2))
            mesh_name = "{}_decimated.off".format(str(num_csv).zfill(2))
            print("======= {} =======".format(file_name))


            #file_name = "mesh{}_rand_man_sim.csv".format(num_csv)
            #mesh_name = "mesh{}_man_sim.off".format(num_csv)
            #center_name = "mesh{}.txt".format(num_csv)

            mesh_file = os.path.join(r"{}\{}\{}".format(*dirs_mesh, mesh_name))
            mesh = openmesh.read_trimesh(mesh_file)
            phi, theta, row = get_row_and_angles_from_mesh(mesh,
                                                           file=mesh_file,
                                                           )
            generator_circle = get_generator_circle_from_cache(row)
            point_cloud = mesh.points()
            normalize(point_cloud)
            # Reorientation
            reorient_point_cloud_by_angles(point_cloud, phi, theta)

            pd_res = pd.read_csv(os.path.join(*dirs_file, file_name))

            add_desviacion(pd_res)

            sample_group = pd_res.groupby("timestamp")
            best_samples = sample_group.nth(-1)

            col_group = "desviacion"
            mean_y_cols = ["normal_x", "normal_y", "normal_z"]
            std_y_cols = ["angle_diff"]
            y_cols = mean_y_cols + std_y_cols

            x, mean_desv_phi, std_ang = calc_mean_std(best_samples,
                                                      [],
                                                      col_group,
                                                      y_cols)

            def min_loss(df):
                return df[df["loss"] == df["loss"].min()]

            a = best_samples.groupby(col_group)
            b = a.apply(min_loss)


            std_ang = std_ang[std_y_cols].to_numpy()

            ps = show_mesh_with_all_found_axes(mesh,
                                               generator_circle,
                                               mean_desv_phi[mean_y_cols].to_numpy(),
                                               ["loss angs {}".format(np.round(i, 2)) for i in x],
                                               [0.0015 for i in x])
                                               #linear_range_map(std_ang, [std_ang.min(), std_ang.max()], [0.0001, 0.001]))
            ps.set_ground_plane_mode("shadow_only")
            ps.show()


if __name__ == "__main__":
    #show_dataset_ps()
    show_known_syms_ps()

