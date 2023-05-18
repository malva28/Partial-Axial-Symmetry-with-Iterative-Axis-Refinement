import os
import csv
import functools
import numpy as np
import pandas as pd
from supporting_circles import Circle
from common_args import create_common_args, symmetry_param_keys


def create_file_in_another_dir(new_parent_dir, base_subdirs, name, clear=False):
    file_dirs = os.path.join(new_parent_dir, base_subdirs)
    file_path = os.path.join(file_dirs, name)

    if not os.path.exists(file_path) or clear:
        os.makedirs(file_dirs, exist_ok=True)
        with open(file_path, "w") as fopen:
            pass

    return file_path


CACHE_NAME = os.path.join("results", "axis", "partial_symmetries.csv")


def cache_exists():
    return os.path.exists(CACHE_NAME)


def cache_empty():
    with open(CACHE_NAME) as csvfile:
        csv_dict = [row for row in csv.DictReader(csvfile)]
        return len(csv_dict) == 0


def read_cache():
    return pd.read_csv(CACHE_NAME)


def find_row_in_cache(df, **parser_vals):
    def and_fun(a, b):
        return a & b

    def val_in_col_check(a):
        return df[a] == parser_vals[a]

    row = df.loc[functools.reduce(and_fun, map(val_in_col_check, parser_vals))]
    i_row = list(df.index[functools.reduce(and_fun, map(val_in_col_check, parser_vals))])
    print("index", i_row, row)
    return row, i_row


def get_generator_circle_from_cache(row):
    generator_circle = Circle(radius=row.iloc[0]["r_circle"],
                              center=np.array([row.iloc[0]["c" + coord + "_circle"] for coord in ["x", "y", "z"]]),
                              normal=np.array([row.iloc[0]["n" + coord + "_circle"] for coord in ["x", "y", "z"]]))
    return generator_circle


def gen_cache_headers():
    headers = [item for item in symmetry_param_keys]
    add_headers = ["r_circle"]
    temp_headers = ["c{}_circle", "n{}_circle"]
    for add_header in temp_headers:
        for coord in ["x", "y", "z"]:
            add_headers.append(add_header.format(coord))
    for angle in ["theta", "phi"]:
        add_headers.append("{}_adjust".format(angle))
    return headers, add_headers


def gen_cache_row(args_dict, headers, generator_circle, phi, theta):
    row = {key: args_dict[key] for key in headers}
    coords = ["x", "y", "z"]
    row["r_circle"] = generator_circle.r
    for i in range(len(coords)):
        coord = coords[i]
        row["c" + coord + "_circle"] = generator_circle.c[i]
    for i in range(len(coords)):
        coord = coords[i]
        row["n" + coord + "_circle"] = generator_circle.n[i]
    row["phi_adjust"] = phi
    row["theta_adjust"] = theta
    return row


def write_cache(args_dict, generator_circle, phi, theta):
    # cache found circle
    cache_file = create_file_in_another_dir(os.path.join("results", "axis"), "", "partial_symmetries.csv")
    bad_headers = cache_empty()
    headers, add_headers = gen_cache_headers()

    if not cache_empty():
        df = read_cache()
        df_cols = list(df.columns.values)
        bad_headers = bad_headers or df_cols != headers + add_headers
    if bad_headers:
        # no header is even present
        row = gen_cache_row(args_dict, headers, generator_circle, phi, theta)
        row = {key: [row[key]] for key in row}
        df = pd.DataFrame(row, columns=headers + add_headers)
        df.to_csv(cache_file, index=False)
    else:
        old_row, i_old_row = find_row_in_cache(df, **{key: args_dict[key] for key in symmetry_param_keys})
        if old_row.empty:
            row = gen_cache_row(args_dict, headers, generator_circle, phi, theta)
            df.loc[len(df)] = row
            df.to_csv(cache_file, index=False)
        else:
            i_old_row = i_old_row[0]
            old_circle = get_generator_circle_from_cache(old_row)
            if not old_circle == generator_circle:
                df.loc[i_old_row] = gen_cache_row(args_dict, headers, generator_circle, phi, theta)
                df.to_csv(cache_file, index=False)






