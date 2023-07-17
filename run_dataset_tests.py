import argparse
import sys
import subprocess
import os
import traceback

import numpy as np

from common_args import create_common_args, symmetry_param_keys, add_shift_args, shift_params_keys


if __name__ == "__main__":
    python_exe = sys.executable
    parser = create_common_args(description='Test iterative shift over dataset meshes')
    add_shift_args(parser)
    parser.add_argument('--cache', default=True, action='store_true', help="True if you want to retrieve initial axis "
                                                                           "from cache file.\nFalse if you want to "
                                                                           "reforce partial axis calculation")
    parser.add_argument('--no-cache', dest='cache', action='store_false')
    parser.add_argument("--num_test",
                        default=20,
                        type=int,
                        help="Number of times to run each test")

    args = parser.parse_args()

    dataset_path = path = os.path.join("files", "Larco")
    off_list = os.listdir(path)
    off_list = [[f for f in off_list if n in f][0] for n in ["18"]]

    key_args = symmetry_param_keys.copy()
    key_args.extend(shift_params_keys)
    key_args.extend(["cache", "visual"])
    arg_dict_list = []
    for key in key_args:
        value = getattr(args, key)
        if type(value) == bool:
            # flag found
            if value:
                arg_dict_list.append("--" + key)
            elif not value and key == "visual":
                arg_dict_list.append("--no-visual")
        else:
            arg_dict_list.append("--" + key)
            arg_dict_list.append(str(value))

    for off_file in off_list:
        off_path = os.path.join(dataset_path, off_file)
        k = arg_dict_list.index("--file")
        arg_dict_list[k+1] = off_path

        print("\n=========================================")
        print("Procesing file: {}".format(off_path))
        rads = np.linspace(0, np.pi/6, num=6, endpoint=False)
        rads = rads + rads[1]

        try:
            for i in range(args.num_test):
                print("\nTest number: {}\n".format(i+1))

                for rad in rads:
                    adjusted_delta_radius = np.tan(rad)
                    print()
                    print("Trying with radius: {}".format(adjusted_delta_radius))

                    j = arg_dict_list.index("--delta_radius")
                    arg_dict_list[j+1] = str(adjusted_delta_radius)

                    subprocess_list = [python_exe, 'main.py']
                    subprocess_list.extend(arg_dict_list)
                    process = subprocess.run(subprocess_list)
                    ret = process.returncode
                    process.check_returncode()
        except Exception as exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(''.join(traceback.format_tb(exception.__traceback__)), file=sys.stderr)
            print(exception, file=sys.stderr)
