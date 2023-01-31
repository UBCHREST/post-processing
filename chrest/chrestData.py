import argparse
import pathlib
import sys
import numpy as np
import h5py

from support.supportPaths import expand_path


class ChrestData:
    # a list of chrest files
    files = None

    # store the files based upon time
    files_per_time = dict()

    # store the list of fields available
    fields = []

    # load the metadata from the first file
    metadata = dict()

    # store the grid information
    start_point = []
    end_point = []
    delta = []
    dimensions = 0
    grid = []

    # store the times available to the data
    times = []

    """
    Creates a new class from hdf5 chrest formatted file(s).
    """

    def __init__(self, files=None):
        if files is None:
            self.files = []
        elif isinstance(files, str):
            self.files = expand_path(files)
        elif isinstance(files, pathlib.Path):
            self.files = expand_path(files)
        else:
            self.files = files

        # Open each file to get the time and check the available fields
        for file in self.files:
            # Load in the hdf5 file
            hdf5 = h5py.File(file, 'r')

            # Check each of the cell fields
            hdf5_data = hdf5['data']
            hdf5_fields = hdf5_data['fields']

            if len(self.fields):
                # check for uniform values
                if self.fields != list(hdf5_fields.keys()):
                    raise Exception("There is an error in the chrest files.  The fields in each file do not match.")
            else:
                # If it is empty just end items
                self.fields = list(hdf5_fields.keys())

                # get the fields and other data
            t = hdf5_data.attrs['time'][0]
            self.files_per_time[t] = file

            # Load in the metadata
            if ~ len(self.metadata):
                for (key, value) in hdf5_data.attrs.items():
                    if h5py.check_string_dtype(value.dtype):
                        self.metadata[key] = value[0]

            # store the grid information, it is assumed to be the same for each file
            hdf5_grid = hdf5_data['grid']
            self.start_point = hdf5_grid['start'][:, 0].tolist()
            self.end_point = hdf5_grid['end'][:, 0].tolist()
            self.delta = hdf5_grid['discretization'][:, 0].tolist()
            self.dimensions = len(self.start_point)

            for dim in range(self.dimensions):
                self.grid.append(int((self.end_point[dim] - self.start_point[dim]) / self.delta[dim]))

        # Store the list of times
        self.times = list(self.files_per_time.keys())
        self.times.sort()

    """
    Create a new grid with the supplied information
    start_point: the start location of the grid
    end_point: the location of the grid
    cells: the number of cells in each direction
    """

    def setup_new_grid(self, start_point, end_point, cells):
        if len(self.start_point):
            raise Exception("A new grid cannot be setup if a current grid exists.")

        self.dimensions = len(start_point)
        self.start_point = start_point
        self.end_point = end_point
        self.grid = cells

        # compute delta
        for dim in range(self.dimensions):
            self.delta.append((self.end_point[dim] - self.start_point[dim]) / self.grid[dim])

    """
    Returns the field data for the specified time range as a list of pairs(time, numpy array).
    The field should be [k, j, i]
    
    """

    def get_field(self, field_name, min_time=-sys.float_info.max, max_time=sys.float_info.max):
        # create a dictionary of times/data
        data = []

        # march over the files
        for t in self.times:
            if min_time <= t <= max_time:
                # Load in the file
                with h5py.File(self.files_per_time[t], 'r') as hdf5:
                    # Check each of the cell fields
                    hdf5_fields = hdf5['data/fields']

                    try:
                        # load in the specified field name
                        hdf5_field = hdf5_fields[field_name]

                        data.append((t, hdf5_field[:]))
                    except Exception as e:
                        raise Exception(
                            "Unable to open field " + field_name + ". Valid fields include: " + ','.join(self.fields))

        return data

    """
    Returns an numpy array of coordinates
    [dim][k][j][i]
    """

    def get_coordinates(self):
        # Note we reverse the order of the linespaces and the returning grid for k,j,i index
        # create a list of line spaces
        linspaces = []
        for dim in range(self.dimensions):
            linspaces.append(
                np.linspace(self.start_point[dim] + self.delta[dim] / 2.0, self.end_point[dim] - self.delta[dim] / 2.0,
                            self.grid[dim], endpoint=True))

        linspaces.reverse()

        # compute the multi dim grid
        grid = np.meshgrid(*linspaces, indexing='ij')
        grid.reverse()
        return grid

    # Save any new data into a new file
    def save(self, path_template):
        # march over the files
        index = 0

        for t in self.times:
            # Name the file
            hdf5_path = path_template + f".{index:05d}.hdf5"

            # Open a new file
            with h5py.File(hdf5_path, 'w') as hdf5:
                # Write in the grid information
                data = hdf5.create_group('data')
                # store the metadata
                for key in self.metadata:
                    data.attrs.create(key, self.metadata[key])
                data.attrs.create('time', t)

                # write the grid information
                grid = data.create_group('grid')
                grid.create_dataset('start', data=np.asarray(self.start_point))
                grid.create_dataset('end', data=np.asarray(self.end_point))
                grid.create_dataset('discretization', data=np.asarray(self.delta))


# parse based upon the supplied inputs
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate an xdmf file from MatLab data files holding structed data. '
                                                 'See https://github.com/cet-lab/experimental-post-processing/wiki'
                                                 '/Matlab-To-XdmfGenerator for details.  ')
    parser.add_argument('--file', dest='hdf5_file', type=pathlib.Path, required=True,
                        help='The path to the hdf5 file(s) containing the structured data.  A wild card can be used '
                             'to supply more than one file.')
    args = parser.parse_args()

    # this is some example code for chest file post processing
    chrest_data = ChrestData(args.hdf5_file)

    # load in an example data
    field_data = chrest_data.get_field("flameTemp")

    # get the coordinate data
    coord_data = chrest_data.get_coordinates()

    # Print off a row of information at each time
    for (time, flame_temp) in field_data:
        i = 0
        j = 0
        k = 0
        for i in range(chrest_data.grid[0]):
            print(time, " @", coord_data[0][k, j, i], ", ", coord_data[1][k, j, i], ", ", coord_data[2][k, j, i], ": ",
                  flame_temp[k, j, i])
