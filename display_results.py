import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


"""
def fun(index, normals, deg_angles, delta_radius, axis_center, mesh, ):
    print("Shifted normal: {} ({} degrees)".format(normals[i, :], angles[i]))
    rad = angles[i] * np.pi / 180
    if test_type == "sanity":
        adjusted_delta_radius = np.tan(rad)
    else:
        adjusted_delta_radius = delta_radius
    used_args["delta_radius"].append(adjusted_delta_radius)
    sym_circle = Circle(center, 1, normals[i, :])
    best_normals, new_generator_circle, num_its = iterative_symmetry_shift(mesh,
                                                                           sym_circle,
                                                                           adjusted_delta_radius,
                                                                           decrease_factor,
                                                                           phi_simmetries,
                                                                           epsilon_radius,
                                                                           chi_convergence,
                                                                           n_sample_points,
                                                                           n_angles)
    best_loss_normal = best_normals[-1, :]
    # resulting_normals.append(np.hstack([num_its, new_generator_circle.n]))
    resulting_normals.append(np.hstack([num_its, best_loss_normal]))
    print("Angle diff with original: {}".format(myangle(new_generator_circle.n, origin_circle.n)))
    if show_ps_results:
        ps_test = show_mesh_with_all_found_axes(mesh, new_generator_circle, best_normals[:, 1:])
        ps_test.register_curve_network(f"Original Symmetry", np.array(
            [-origin_circle.n + origin_circle.c, origin_circle.n + origin_circle.c]), np.array([[0, 1]]),
                                       radius=0.002)
        ps_test.show()
"""

if __name__ == "__main__":
    num_csv = 7
    file_name = "mesh{}_man_sim.csv".format(num_csv)
    pd_res = pd.read_csv(os.path.join("results", "axis", file_name))

    # group_by_loss_angles = pd_res.groupby("phi_simmetries")
    #group_by_loss_angles = pd_res.loc[pd_res["chi_convergence"] == 0.0001].loc[pd_res["phi_simmetries"] == 16].groupby(["n_angles", "desviacion"])
    #group_by_loss_angles = pd_res.loc[pd_res["chi_convergence"] == 0.0001].loc[pd_res["phi_simmetries"] == 16].loc[pd_res["n_angles"] > 1].groupby(
    #    ["desviacion", "n_angles"])
    group_by_loss_angles = pd_res.loc[pd_res["chi_convergence"] == 0.0001].loc[pd_res["phi_simmetries"] == 16].loc[
        pd_res["n_angles"] > 1].groupby(["n_angles", "desviacion"])
    #group_by_loss_angles = pd_res.loc[pd_res["chi_convergence"] == 0.0001].groupby("n_angles")
    mean_desv_phi = group_by_loss_angles.mean()
    indices = mean_desv_phi.index.get_level_values(0).unique()

    for i in indices:
        mean_i = mean_desv_phi.loc[[i], ["loss", "angle_diff"]]
        x = mean_i.index.get_level_values(1)
        std_i = group_by_loss_angles.std().loc[[i], ["loss", "angle_diff"]]
        # https://stackoverflow.com/questions/62177520/how-to-add-error-bars-in-matplotlib-for-multiple-groups-from-dataframe
        fig, ax = plt.subplots()
        for col in ["loss", "angle_diff"]:
            ax.errorbar(x,
                        mean_i[col],
                        std_i[col],
                        label=col)
        plt.title("val {}".format(i))

        plt.show()
        print("hola")



    """
    for i in range(6):
        res = group_by_loss_angles.nth(i).loc[2:, ["loss", "angle_diff"]]
        res.plot()
        plt.show()
        n_rows = len(res)

    
    """
    print(group_by_loss_angles)