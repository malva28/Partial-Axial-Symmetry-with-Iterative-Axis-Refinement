import pandas as pd
import os
import sys
import subprocess
import functools
import numpy as np
import openmesh

from common_args import create_common_args, symmetry_param_keys
from supporting_circles import compute_supporting_circles, Circle
from mesh_axis_render import show_mesh_with_partial_axis, take_screenshots
from utils import cache_exists, cache_empty, read_cache, find_row_in_cache, get_generator_circle_from_cache


def use_from_cache(parser_vals):
    print("Using cached values for symmetry axis")
    df = read_cache()

    read_cache()
    row,_ = find_row_in_cache(df, **parser_vals)

    generator_circle = get_generator_circle_from_cache(row)
    phi_adjust = row.iloc[0]["phi_adjust"]
    theta_adjust = row.iloc[0]["theta_adjust"]
    print(generator_circle)
    print("Reading Mesh")
    mesh = openmesh.read_trimesh(args.file)

    show_mesh_with_partial_axis(mesh, generator_circle, args.symmetric_support_threshold, phi_adjust, theta_adjust)
    return generator_circle



def run_initial_partial_symmetry():
    venv_exe = sys.executable
    flat_args = []
    for key in symmetry_param_keys:
        flat_args.append("--" + key)
        flat_args.append(str(arg_dict[key]))
    process = subprocess.run([venv_exe, 'main_partial_symmetry.py', *flat_args], check=True)
    ret = process.returncode
    print("Finished executing main_partial_symmetry with return code {}".format(ret))
    return ret


if __name__ == "__main__":
    parser = create_common_args(description='Mesh signature visualization')
    parser.add_argument('--cache', default=True, action='store_true', help="True if you want to retrieve initial axis "
                                                                           "from cache file.\nFalse if you want to "
                                                                           "reforce partial axis calculation")
    parser.add_argument('--no-cache', dest='cache', action='store_false')

    args = parser.parse_args()
    arg_dict = vars(args)
    parser_vals = {key: arg_dict[key] for key in symmetry_param_keys}

    not_cached = False

    if args.cache:
        exists_ = cache_exists()
        not_cached = not cache_exists() or cache_empty() or find_row_in_cache(read_cache(), **parser_vals)[0].empty

    not_cached = not args.cache or not_cached

    if not_cached:
        ret = run_initial_partial_symmetry()
        df = read_cache()
        row, _ = find_row_in_cache(df, **parser_vals)
        generator_circle = get_generator_circle_from_cache(row)
    else:
        generator_circle = use_from_cache(parser_vals)




