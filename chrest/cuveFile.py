import argparse
import pathlib
import sys

import numpy as np
import h5py

from chrest.ablateData import AblateData
from chrestData import ChrestData
from supportPaths import expand_path
from scipy.spatial import KDTree

# parse based upon the supplied inputs
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate curve files from ablate files')
    parser.add_argument('--file', dest='file', type=pathlib.Path, required=True,
                        help='The path to the ablate hdf5 file(s) containing the ablate data.'
                             '  A wild card can be used '
                             'to supply more than one file.')

    args = parser.parse_args()

    # this is some example code for chest file post-processing
    ablate_data = AblateData(args.file)

    fields = ablate_data.get_fields()

    # create an output directory
    information_name = str(args.file.stem).replace("*", "")
    curve_path_base = args.file.parent / (information_name + "curve")
    curve_path_base.mkdir(parents=True, exist_ok=True)
    curve_path_base = curve_path_base / (information_name + "curve")

    # make sure that the mesh is 1D
    cell_centers = ablate_data.compute_cell_centers()
    if len(cell_centers.shape) != 2 and cell_centers.shape[1] != 1:
        raise Exception("The input data must be 1D to be converted to a curve file")

    # flatten the data
    cell_centers = cell_centers[:, 0]

    # create a new file for each field based upon time
    file_index = 0
    for time in ablate_data.times:
        curve_path = str(curve_path_base) + f".{file_index:05d}.curve"
        with open(curve_path, 'w') as f:
            f.write(f'#title={information_name}\n')
            f.write(f'##time={time}\n')

        # bump the index
        file_index += 1

    # Now output each field
    for field in fields:
        # get the field from ablate
        data, number_components, component_names = ablate_data.get_field(field)

        # write each time step
        file_index = 0
        for time in ablate_data.times:
            curve_path = str(curve_path_base) + f".{file_index:05d}.curve"
            with open(curve_path, 'a') as f:

                # if there is only one component
                if number_components <= 1:
                    # write the name component
                    f.write(f'\n#{field}\n')

                    # get the data at this time step
                    data_at_timestep = data[file_index, :]

                    for i in range(len(data_at_timestep)):
                        f.write(f'{cell_centers[i]} {data_at_timestep[i]}\n')
                # else there are more than one component
                else:
                    for c in range(number_components):
                        # write the name component
                        f.write(f'\n#{field}_{component_names[c]}\n')

                        # get the data at this time step
                        data_at_timestep = data[file_index, :, c]

                        for i in range(len(data_at_timestep)):
                            f.write(f'{cell_centers[i]} {data_at_timestep[i]}\n')
            # bump the index
            file_index += 1
