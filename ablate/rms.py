import argparse
import math
import pathlib

import h5py
import numpy

from chrest.supportPaths import expand_path


def find_cell_index(start, dx, search):
    offset = search - start
    return math.floor(offset / dx)


# parse based upon the supplied inputs
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Takes the mean and the rms')
    parser.add_argument('--file', dest='hdf5_file', type=pathlib.Path, required=True,
                        help='The input hdf5_file')
    parser.add_argument('--field', dest='field', type=str, required=False,
                        help='The name of the field', default="cell_fields/aux_temperature")
    args = parser.parse_args()

    # expand any wild cards in the path name
    hdf5_paths = expand_path(args.hdf5_file)

    # Create the output file
    output_file = args.hdf5_file.parent / "stats.h5"
    hdf5_dest = h5py.File(output_file, 'w')

    # hold the main data
    initiated = False

    # get the fields
    field_avg = None
    field_rms = None

    # march over each file with path
    count = 0
    for hdf5_file in hdf5_paths:
        print("Opening ", hdf5_file)
        count += 1
        # Load in the hdf5 file
        hdf5_source = h5py.File(hdf5_file, 'r')

        # get the base field to get the size
        source_field = hdf5_source[args.field]
        # set up the output on the first time
        if not initiated:
            initiated = True
            hdf5_source.copy(hdf5_source["viz"], hdf5_dest, "viz")
            hdf5_source.copy(hdf5_source["topology"], hdf5_dest, "topology")
            hdf5_source.copy(hdf5_source["time"], hdf5_dest, "time")
            hdf5_source.copy(hdf5_source["geometry"], hdf5_dest, "geometry")

            # set up default values
            field_avg = hdf5_dest.create_dataset(args.field + "_avg", source_field.shape, 'f',
                                                 fillvalue=0)
            field_avg.attrs.update(source_field.attrs)
            field_rms = hdf5_dest.create_dataset(args.field + "_rms", source_field.shape, 'f',
                                                 fillvalue=0)
            field_rms.attrs.update(source_field.attrs)

        field_avg[:] = numpy.add(field_avg[:], source_field[:])
        field_rms[:] = numpy.add(field_rms[:], numpy.square(source_field[:]))
        hdf5_source.close()

        print("Added in contribution from ", hdf5_file)

    # Now finish
    field_avg[:] = field_avg[:] / count
    field_rms[:] = field_rms[:] / count - numpy.square(field_avg[:])
    field_rms[:] = numpy.maximum(field_rms[:], 0)
    field_rms[:] = numpy.sqrt(field_rms[:])

    hdf5_dest.close()
