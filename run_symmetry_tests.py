import argparse
import sys
import subprocess
import os

from common_args import add_shift_args, shift_params_keys

if __name__ == "__main__":
    python_exe = sys.executable
    parser = argparse.ArgumentParser(description='Test iterative shift over meshes that have known symmetries')
    add_shift_args(parser)
    parser.add_argument("--test_type",
                        default='normal',
                        choices=["normal", "sanity"],
                        type=str,
                        help='File to use')
    parser.add_argument('--show_ps_results',
                        default=False,
                        action='store_true',
                        help="True if you want to display results.")
    parser.add_argument("--plt_y_axis",
                        default='loss',
                        choices=["loss", "iteration"],
                        type=str,
                        help='File to use')
    parser.add_argument("--num_test",
                        default=20,
                        type=int,
                        help="Number of times to run each test")

    args = parser.parse_args()

    key_args = shift_params_keys.copy()
    key_args.extend(["test_type", "show_ps_results", "plt_y_axis"])
    arg_dict_list = []
    for key in key_args:
        value = getattr(args, key)
        if type(value) == bool:
            # flag found
            if value:
                arg_dict_list.append("--" + key)
        else:
            arg_dict_list.append("--" + key)
            arg_dict_list.append(str(value))

    for num_test in [1, 2, 4, 5, 6, 7]:
        path_axes = "files/sym_test/mesh{}.txt".format(num_test)
        path_mesh = "files/sym_test/mesh{}_man_sim.off".format(num_test)

        print("\n=========================================")
        print("Procesing file: {}".format(path_mesh))

        for i in range(args.num_test):
            print("\nTest number: {}\n".format(i+1))
            subprocess_list = [python_exe, 'symmetry_test.py',
                               "--axis_file", path_axes,
                               "--file", path_mesh]
            subprocess_list.extend(arg_dict_list)
            process = subprocess.run(subprocess_list)
            ret = process.returncode


