import argparse
import math
import pathlib
import time

import h5py
import numpy

from support.matlabToXdmfGenerator import generate_xdmf


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
    args = parser.parse_args()

    # Load in the hdf5 file
    hdf5 = h5py.File(args.hdf5_file, 'r')

    # find the max, min x,y,z values
    vertices = hdf5["geometry"]["vertices"]

    x_coord = vertices[:, 0]
    y_coord = vertices[:, 1]
    maxCoord = [numpy.amax(x_coord), numpy.amax(y_coord)]
    minCoord = [numpy.amin(x_coord), numpy.amin(y_coord)]

    # create a result size
    output_file = args.hdf5_file.parent / (args.hdf5_file.stem + ".project.hdf5")
    project_hdf5_root = h5py.File(output_file, 'w')

    # copy over the known info
    start_dataset = project_hdf5_root.create_dataset("main/start", (2), 'f')
    start_dataset[:] = [minCoord[0], minCoord[1]]
    discretization_dataset = project_hdf5_root.create_dataset("main/discretization", (2), 'f')
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
            field_dataset_avg[j, i] /= field_count[j, i]
            if field_dataset_min[j, i] < 1E-8:
                field_dataset_avg[j, i] = 0
                field_dataset_max[j, i] = 0

    project_hdf5_root.close()

    # Create the xdmf file
    xdmf_file = output_file.parent / (output_file.stem + ".xdmf")
    generate_xdmf(output_file, xdmf_file)
