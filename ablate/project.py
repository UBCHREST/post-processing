import argparse
import math
import pathlib
import numpy
from mpi4py import MPI
import h5py
import time
import sys


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


def set_cell_ranges(inner_comm, total_num_cells):
    inner_rank = inner_comm.Get_rank()
    inner_size = inner_comm.Get_size()

    local_num = int(total_num_cells / inner_size)
    if inner_rank == 0:
        local_num += total_num_cells % inner_size

    # Split up all cells here
    buf = numpy.empty(1, dtype=int)
    if inner_rank == 0:
        low = 0
        high = local_num - 1
        if inner_size > 1:
            buf[0] = high + 1
            comm.Send(buf, 1)
    else:
        comm.Recv(buf, rank - 1)
        low = buf[0]
        high = low + local_num - 1
        if inner_rank < inner_size - 1:
            buf[0] = high + 1
            comm.Send(buf, rank + 1)

    return [low, high], local_num


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

    # Load in the comm, size, and rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Load in all HDF5 files here
    # Need to have all files visible to all processes, so this is fine
    [all_hdf5_files, multi_check] = get_all_files(args.hdf5_file)

    # If we want a new directory, then create one here. This should only be done by one process
    if args.newDir and args.hdf5_file.is_dir() and multi_check:
        output_dir = pathlib.Path(str(args.hdf5_file) + '/' + args.hdf5_file.name + '-projections')
        if not output_dir.exists() and rank == 0:
            output_dir.mkdir()

    # Loop over all hdf5 files
    local_vertex_range = 0
    local_cell_range = 0
    local_cleanup_range = 0

    local_num_cells = 0

    for hdf5_file in all_hdf5_files:
        if rank == 0:
            start_time = time.time()
            print(hdf5_file.name)

        # We need to make this parallel read, because we don't need multiple iterations of this file opened elsewhere
        hdf5 = h5py.File(hdf5_file, 'r', driver='mpio', comm=MPI.COMM_WORLD)

        # find the max, min x,y,z values
        cells = hdf5["viz"]["topology"]["cells"]
        field3D = hdf5["cell_fields"][args.field]
        vertices = hdf5["geometry"]["vertices"]

        total_cells_all = cells.shape[0]
        if type(local_cell_range) != list:
            local_cell_range, local_num_cells = set_cell_ranges(comm, total_cells_all)

        # create a result size
        if args.newDir and args.hdf5_file.is_dir() and multi_check:
            output_file = output_dir / (hdf5_file.stem + ".project.hdf5")
        else:
            output_file = hdf5_file.parent / (hdf5_file.stem + ".project.hdf5")
        project_hdf5_root = h5py.File(output_file, 'w', driver='mpio', comm=MPI.COMM_WORLD)

        # copy over the known info
        start_dataset = project_hdf5_root.create_dataset("main/start", (2,), 'f')
        discretization_dataset = project_hdf5_root.create_dataset("main/discretization", (2,), 'f')

        if type(local_vertex_range) != list:
            local_vertex_range, _ = set_cell_ranges(comm, len(vertices))

        x_coord = vertices[local_vertex_range[0]:(local_vertex_range[1]+1), 0]
        y_coord = vertices[local_vertex_range[0]:(local_vertex_range[1]+1), 1]
        loc_min = numpy.array([numpy.amin(x_coord), numpy.amin(y_coord)])
        loc_max = numpy.array([numpy.amax(x_coord), numpy.amax(y_coord)])

        maxCoord = numpy.zeros(2)
        minCoord = numpy.zeros(2)
        comm.Reduce(loc_min, minCoord, op=MPI.MIN, root=0)
        comm.Reduce(loc_max, maxCoord, op=MPI.MAX, root=0)

        if rank == 0:
            start_dataset[:] = [minCoord[0], minCoord[1]]
            discretization_dataset[:] = [(maxCoord[0] - minCoord[0]) / args.nx, (maxCoord[1] - minCoord[1]) / args.ny]

        # size up the field
        # These operations need to be collective. These are acceptable
        field_name = "temperatureRMS"
        field_dataset_avg = project_hdf5_root.create_dataset("main/" + field_name + "_avg", (args.ny, args.nx), 'f',
                                                             fillvalue=0)
        field_dataset_max = project_hdf5_root.create_dataset("main/" + field_name + "_max", (args.ny, args.nx), 'f',
                                                             fillvalue=-1000000)
        field_dataset_min = project_hdf5_root.create_dataset("main/" + field_name + "_min", (args.ny, args.nx), 'f',
                                                             fillvalue=1000000)
        field_count = project_hdf5_root.create_dataset("main/count", (args.ny, args.nx), 'f')

        # Find the correct component
        component_index = -1
        if len(field3D.shape) == 3:
            number_components = field3D.shape[2]
        else:
            number_components = 1

        buffer = numpy.empty(1, dtype=int)
        if len(field3D.shape) == 3:
            if rank == 0:
                for c in range(number_components):
                    attribute_name_test = "componentName" + str(c)
                    if field3D.attrs[attribute_name_test].decode("utf-8") == args.component:
                        component_index = c
                buffer[0] = component_index
                comm.Bcast(buffer, 0)
            else:
                comm.Bcast(buffer, 0)
                component_index = buffer[0]

        # Now march over each 3d cell
        for c in range(local_cell_range[0], local_cell_range[1] + 1):
            # get the field value at this cell
            if len(field3D.shape) == 3:
                field_value = field3D[0, c, component_index]
            else:
                field_value = field3D[0, c]
            # convert from rms
            # hard code bad
            field_value = field_value / 6881.0

            # get the center of the cell
            cell_center = [0, 0]
            cell_vertices = numpy.take(vertices, cells[c], axis=0)

            # get the cell i, j
            cell_center = numpy.sum(cell_vertices, axis=0)
            i = find_cell_index(start_dataset[0], discretization_dataset[0], cell_center[0] / len(cell_vertices))
            j = find_cell_index(start_dataset[1], discretization_dataset[1], cell_center[1] / len(cell_vertices))

            # take the average for now
            field_dataset_max[j, i] = max(field_value, field_dataset_max[j, i])
            field_dataset_min[j, i] = min(field_value, field_dataset_min[j, i])
            field_dataset_avg[j, i] += field_value
            field_count[j, i] += 1

            # check for output
            if c % 1000 == 0 and rank == 0:
                print((c / local_num_cells) * 100, "% done...")
                sys.stdout.flush()

        # Cleanup!
        if type(local_cleanup_range) != list:
            total_cleanup_cells = args.nx * args.ny
            local_cleanup_range, _ = set_cell_ranges(comm, total_cleanup_cells)

        for cell_num in range(local_cleanup_range[0], local_cleanup_range[1]):
            i = int(cell_num / args.ny)
            j = cell_num % args.ny
            if field_count[j, i] != 0:
                field_dataset_avg[j, i] /= field_count[j, i]
            if field_dataset_min[j, i] < 1E-8:
                field_dataset_avg[j, i] = 0
                field_dataset_max[j, i] = 0

        project_hdf5_root.close()
        hdf5.close()

        if rank == 0:
            end_time = time.time()
            print(end_time - start_time, " s")

    if rank == 0:
        print("Done!")
