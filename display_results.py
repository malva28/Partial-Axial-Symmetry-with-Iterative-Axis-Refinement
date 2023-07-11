import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import openmesh
from supporting_circles import Circle
from iterative_shift import show_mesh_with_all_found_axes
from symmetry_test import parse_axis_perturbation_file
from transformations import rad_to_degree, linear_range_map


def get_cmap(n, name='hsv'):
    """
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.
    reference: https://matplotlib.org/1.2.1/mpl_examples/pylab_examples/show_colormaps.pdf
    """
    return plt.cm.get_cmap(name, n)


if __name__ == "__main__":
    num_csv = 7
    file_name = "mesh{}_man_sim.csv".format(num_csv)
    mesh_name = "mesh{}_man_sim.off".format(num_csv)
    center_name = "mesh{}.txt".format(num_csv)

    pd_res = pd.read_csv(os.path.join("results", "axis", file_name))
    pd_res["angle_diff"] = rad_to_degree(pd_res["angle_diff"])
    # group_by_loss_angles = pd_res.groupby("phi_simmetries")
    # group_by_loss_angles = pd_res.loc[pd_res["chi_convergence"] == 0.0001].loc[pd_res["phi_simmetries"] == 16].groupby(["n_angles", "desviacion"])
    # group_by_loss_angles = pd_res.loc[pd_res["chi_convergence"] == 0.0001].loc[pd_res["phi_simmetries"] == 16].loc[pd_res["n_angles"] > 1].groupby(
    #    ["desviacion", "n_angles"])
    group_by_loss_angles = pd_res.loc[pd_res["chi_convergence"] == 0.0001].loc[pd_res["phi_simmetries"] == 16].loc[
        pd_res["n_angles"] > 1].groupby(["n_angles", "desviacion"])
    # group_by_loss_angles = pd_res.loc[pd_res["chi_convergence"] == 0.0001].groupby("n_angles")
    mean_desv_phi = group_by_loss_angles.mean()
    indices = mean_desv_phi.index.get_level_values(0).unique()

    for i in indices:
        shown_y_columns = ["iteration", "angle_diff"]
        mean_i = mean_desv_phi.loc[[i], shown_y_columns]
        x = mean_i.index.get_level_values(1)
        std_i = group_by_loss_angles.std().loc[[i], shown_y_columns]
        # https://stackoverflow.com/questions/62177520/how-to-add-error-bars-in-matplotlib-for-multiple-groups-from-dataframe
        fig, ax1 = plt.subplots()
        m = len(shown_y_columns)
        cmap = get_cmap(m + 1, "brg")
        lines = []
        for j in range(m):
            col = shown_y_columns[j]
            if j != 0:
                ax = ax1.twinx()
            else:
                ax = ax1
            # source:
            # https://stackoverflow.com/questions/58009069/how-to-avoid-overlapping-error-bars-in-matplotlib
            lines.append(ax.errorbar(x,
                                     mean_i[col],
                                     std_i[col],
                                     label=col,
                                     color=cmap(j),
                                     alpha=.85,
                                     fmt=':',
                                     capsize=3,
                                     capthick=1))
            ax.set_ylabel(col)
            ax.fill_between(x,
                            mean_i[col]-std_i[col],
                            mean_i[col]+std_i[col],
                            color=cmap(j),
                            alpha=.25)
        labs = [l.get_label() for l in lines]
        ax1.legend(lines, labs, loc=0)
        # fig.tight_layout()
        plt.title("val {}".format(i))

        plt.show()
        print("hola")

        std_ang = std_i["angle_diff"].to_numpy()
        center, normals, _ = parse_axis_perturbation_file(os.path.join("files", "sym_test", center_name))
        mesh = openmesh.read_trimesh(os.path.join("files", "sym_test", mesh_name))
        ps = show_mesh_with_all_found_axes(mesh,
                                           Circle(center, 1, normals[0, :]),
                                           mean_desv_phi.loc[[i], ["normal_x", "normal_y", "normal_z"]].to_numpy(),
                                           ["loss angs {}".format(i) for i in x],
                                           linear_range_map(std_ang, [std_ang.min(), std_ang.max()], [0.0001, 0.001]))
        ps.show()
