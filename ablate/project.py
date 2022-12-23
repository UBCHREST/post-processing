import argparse
import math
import pathlib

import h5py
import numpy


# from support.matlabToXdmfGenerator import generate_xdmf

# Need a function that can get all the hdf5 file paths into a list
def get_all_files(files):
    # First, check if the "directory" is actually a directory
    if files.is_dir():
        # Search through the directory for files fitting the desired pattern
        name_dir = files.name
        pattern = name_dir + '.*.hdf5'
        all_files = sorted(files.glob(pattern))

        # Make sure there are at most only 2 suffixes per filename
        delete_file_ref = []
        for f in range(0, len(all_files)):
            curr_suffix = all_files[f].suffixes
            if len(curr_suffix) > 2:
                delete_file_ref.append(f)

        # Delete erroneous suffixes from list
        delete_file_ref.reverse()
        for f in delete_file_ref:
            all_files.pop(f)

        # Raise exception if the list is now empty
        if len(all_files) == 0:
            raise Exception("There are no valid files in this directory!")

        # Bring in multi-file check
        multi_file = True

    elif files.is_file():
        test_suffixes = files.suffixes
        if ".hdf5" in test_suffixes and len(test_suffixes) <= 2:
            if len(test_suffixes) == 2:
                multi_file = True
            else:
                multi_file = False
            all_files = [files]
        else:
            raise Exception("This file is not valid")
    else:
        raise Exception("Not a valid file location")

    return [all_files, multi_file]


def find_cell_index(start, dx, search):
    offset = search - start
    return math.floor(offset / dx)


# parse based upon the supplied inputs
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Projects the field onto the zPlane based upon a set math operation')
    parser.add_argument('--file', dest='hdf5_file', type=pathlib.Path, required=True,
                        help='The input hdf5_file')
    parser.add_argument('--nx', dest='nx', type=int, required=False,
                        help='The number of x coordinates', default=336)
    parser.add_argument('--ny', dest='ny', type=int, required=False,
                        help='The number of y coordinates', default=48)
    parser.add_argument('--field', dest='field', type=str, required=False,
                        help='The name of the field', default="monitor_temperature")
    parser.add_argument('--component', dest='component', type=str, required=False,
                        help='The name of the component in the field', default="sum")
    parser.add_argument('--new_dir', dest='newDir', type=bool, required=False,
                        help='Controls whether there is a new directory or not', default=False)
    args = parser.parse_args()

    # Construction zone! Beep beep beep!
    [all_hdf5_files, multi_check] = get_all_files(args.hdf5_file)

    # If we want a new directory, then create one here
    if args.newDir and args.hdf5_file.is_dir() and multi_check:
        output_dir = pathlib.Path(str(args.hdf5_file) + '/' + args.hdf5_file.name + '-projections')
        if not output_dir.exists():
            output_dir.mkdir()

    # Loop over all hdf5 files
    for hdf5_file in all_hdf5_files:
        print(hdf5_file.name)
        hdf5 = h5py.File(hdf5_file, 'r')

        # find the max, min x,y,z values
        vertices = hdf5["geometry"]["vertices"]

        x_coord = vertices[:, 0]
        y_coord = vertices[:, 1]
        maxCoord = [numpy.amax(x_coord), numpy.amax(y_coord)]
        minCoord = [numpy.amin(x_coord), numpy.amin(y_coord)]

        # create a result size
        if args.newDir and args.hdf5_file.is_dir() and multi_check:
            output_file = output_dir / (hdf5_file.stem + ".project.hdf5")
        else:
            output_file = hdf5_file.parent / (hdf5_file.stem + ".project.hdf5")
        project_hdf5_root = h5py.File(output_file, 'w')

        # copy over the known info
        start_dataset = project_hdf5_root.create_dataset("main/start", (2,), 'f')
        start_dataset[:] = [minCoord[0], minCoord[1]]
        discretization_dataset = project_hdf5_root.create_dataset("main/discretization", (2,), 'f')
        discretization_dataset[:] = [(maxCoord[0] - minCoord[0]) / args.nx, (maxCoord[1] - minCoord[1]) / args.ny]

        # size up the field
        field_name = "temperatureRMS"
        field_dataset_avg = project_hdf5_root.create_dataset("main/" + field_name + "_avg", (args.ny, args.nx), 'f',
                                                             fillvalue=0)
        field_dataset_max = project_hdf5_root.create_dataset("main/" + field_name + "_max", (args.ny, args.nx), 'f',
                                                             fillvalue=-1000000)
        field_dataset_min = project_hdf5_root.create_dataset("main/" + field_name + "_min", (args.ny, args.nx), 'f',
                                                             fillvalue=1000000)
        field_count = project_hdf5_root.create_dataset("main/count", (args.ny, args.nx), 'f')

        # Now march over each 3d cell
        cells = hdf5["viz"]["topology"]["cells"]
        field3D = hdf5["cell_fields"][args.field]

        # find the component index for name
        number_components = field3D.shape[2]
        component_index = -1
        coords = vertices[:]
        for c in range(number_components):
            attribute_name_test = "componentName" + str(c)
            if field3D.attrs[attribute_name_test].decode("utf-8") == args.component:
                component_index = c

        field_values = field3D[0, :, component_index]
        total_cells = len(cells[:])
        for c in range(total_cells):
            # get the field value at this cell
            field_value = field_values[c]
            # convert from rms
            # hard code bad
            field_value = field_value / 6881.0
            #
            # # get the center of the cell
            cell_center = [0, 0]
            # nodes = numpy.sort(cells[c])
            cell_vertices = coords.take(cells[c], axis=0)
            #
            # # get the cell i, j
            cell_center = numpy.sum(cell_vertices, axis=0)
            i = find_cell_index(start_dataset[0], discretization_dataset[0], cell_center[0] / len(cell_vertices))
            j = find_cell_index(start_dataset[1], discretization_dataset[1], cell_center[1] / len(cell_vertices))

            # # take the average for now
            field_dataset_max[j, i] = max(field_value, field_dataset_max[j, i])
            field_dataset_min[j, i] = min(field_value, field_dataset_min[j, i])
            field_dataset_avg[j, i] += field_value
            field_count[j, i] += 1

            # check for output
            if c % 1000 == 0:
                print((c / total_cells) * 100, "% done...")

        # now cleanup
        for i in range(args.nx):
            for j in range(args.ny):
                if field_count[j, i] != 0:
                    field_dataset_avg[j, i] /= field_count[j, i]
                if field_dataset_min[j, i] < 1E-8:
                    field_dataset_avg[j, i] = 0
                    field_dataset_max[j, i] = 0

        project_hdf5_root.close()

    print("Done!")
    # This section does NOT work for some reason
    # Create the xdmf file
    # if not multi_check:
    #     xdmf_file = output_file.parent / (output_file.stem + ".xdmf")
    #     generate_xdmf(output_file, xdmf_file)
