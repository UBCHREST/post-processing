import argparse
import math
import pathlib

import numpy as np
import pandas as pd

ignore_variable_list = ['avtOriginalNodeNumbers', 'avtOriginalCellNumbers']

def parse_curve_file(path):
    # Using readline()
    curve_file = open(path, 'r')

    # keep active variable name
    variableName = None
    currentTime = np.nan

    data_frame = pd.DataFrame()

    while True:
        # Get next line from file
        line = curve_file.readline()

        # if line is empty
        # end of file is reached
        if not line:
            break

        # check if this is a comment line
        if line.startswith("# TIME"):
            # remove the prefix
            line = line.removeprefix("# TIME")
            currentTime = float(line)

        elif line.startswith("#"):
            line = line.removeprefix("#")
            variableName = line.strip()

            # check if the variable name is in the ignore list
            if variableName in ignore_variable_list:
                variableName = None

        elif len(line) > 0 and variableName:
            data = line.split()
            if len(data) != 2:
                raise Exception("malformed data in curve file")
            else:
                data_frame.at[int(data[0]), variableName] = float(data[1])

        # Add time
        if not math.isnan(currentTime):
            data_frame['time'] = currentTime

    curve_file.close()
    return data_frame


# parse based upon the supplied inputs
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple executable to read in a curve file and convrt to a csv')
    parser.add_argument('--file', dest='curve_file', type=pathlib.Path, required=True,
                        help='A simple curve file, or list of curve file')
    args = parser.parse_args()

    # print all info
    pd.set_option('display.max_columns', None)

    # expand any wild cards in the path name
    # hdf5_paths = expand_path(args.hdf5_file)

    # convert with path
    # for hdf5_file in hdf5_paths:
    # create component markdown
    df = parse_curve_file(args.curve_file)

    csv_path = args.curve_file.parent / (args.curve_file.stem + ".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path)
