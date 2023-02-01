import argparse
import pathlib
import sys
import numpy as np
import h5py

from chrest.xdmfGenerator import XdmfGenerator
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

    # store any newly created data
    new_data = dict()

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
            tt = hdf5_data.attrs['time'][0]
            self.files_per_time[tt] = file

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
            self.grid = []

            for dim in range(self.dimensions):
                self.grid.append(int((self.end_point[dim] - self.start_point[dim]) / self.delta[dim]))

            # if the dim is less than 3, add addition components
            for d in range(self.dimensions, 3):
                self.start_point.append(0)
                self.end_point.append(0)
                self.grid.append(1)
                self.delta.append(0)

        # Store the list of times
        self.times = list(self.files_per_time.keys())
        self.times.sort()

    """
    Create a new grid with the supplied information
    start_point: the start location of the grid
    end_point: the location of the grid
    cells: the number of cells in each direction
    """

    def setup_new_grid(self, start_point, end_point, delta):
        if len(self.start_point):
            raise Exception("A new grid cannot be setup if a current grid exists.")

        self.dimensions = len(start_point)
        self.start_point = start_point
        self.end_point = end_point
        self.delta = delta

        # compute delta
        for dim in range(self.dimensions):
            self.grid.append(int((self.end_point[dim] - self.start_point[dim]) / self.delta[dim]))

        # if the dim is less than 3, add addition components
        for d in range(self.dimensions, 3):
            self.start_point.append(0)
            self.end_point.append(0)
            self.grid.append(1)
            self.delta.append(0)

    """
    Returns the field data for the specified time range as a single (numpy array) and times.
    The field should be [t, k, j, i]
    
    """

    def get_field(self, field_name, min_time=-sys.float_info.max, max_time=sys.float_info.max):
        # create a dictionary of times/data
        data = []
        field_times = []
        # march over the files
        for tt in self.times:
            if min_time <= tt <= max_time:
                field_times.append(tt)
                # Load in the file
                with h5py.File(self.files_per_time[tt], 'r') as hdf5:
                    # Check each of the cell fields
                    hdf5_fields = hdf5['data/fields']

                    try:
                        # load in the specified field name
                        hdf5_field = hdf5_fields[field_name]

                        # prefix any required dummy axes (like k in 2d)
                        hdf5_field_data = hdf5_field[:]

                        for axis in range(self.dimensions, len(self.grid)):
                            hdf5_field_data = np.expand_dims(hdf5_field_data, axis=0)

                        data.append(hdf5_field_data)
                    except Exception as e:
                        raise Exception(
                            "Unable to open field " + field_name + ". Valid fields include: " + ','.join(
                                self.fields) + ". " + str(e))

        return np.stack(data), field_times

    """
    Creates and returns a new field (numpy array) and times.
    The field should be [t, k, j, i, c]

    """

    def create_field(self, field_name, number_components = 1):
        # determine the shape
        shape = [len(self.times)]

        # add in each grid, note the reverse for k, j, i formatting
        grid = self.grid.copy()
        grid.reverse()
        shape.extend(grid)
        # this is where any components would be added
        shape.extend(number_components)

        # create the new field
        data = np.zeros(tuple(shape))

        # store it for future output
        self.new_data[field_name] = data

        return data, self.times

    """
    Returns an numpy array of coordinates
    [k, j, i, dim]
    """

    def get_coordinates(self):
        # Note we reverse the order of the linespaces and the returning grid for k,j,i index
        # create a list of line spaces
        linspaces = []
        for dim in range(len(self.grid)):
            linspaces.append(
                np.linspace(self.start_point[dim] + self.delta[dim] / 2.0, self.end_point[dim] - self.delta[dim] / 2.0,
                            self.grid[dim], endpoint=True))

        linspaces.reverse()

        # compute the multi dim grid
        grid = np.meshgrid(*linspaces, indexing='ij')
        grid.reverse()
        return np.stack(grid, axis=-1)

    # Save any new data into a new file
    def save(self, path_template):
        # generate an xdfm object at the same time
        xdfm = XdmfGenerator()

        # march over the files
        for index in range(len(self.times)):
            # Name the file
            hdf5_path = str(path_template) + f".{index:05d}.hdf5"

            # Open a new file
            with h5py.File(hdf5_path, 'w') as hdf5:
                # Write in the grid information
                data = hdf5.create_group('data')
                # store the metadata
                for key in self.metadata:
                    data.attrs.create(key, self.metadata[key])
                data.attrs.create('time', self.times[index])

                # write the grid information
                grid = data.create_group('grid')
                grid.create_dataset('start', data=np.asarray(self.start_point))
                grid.create_dataset('end', data=np.asarray(self.end_point))
                grid.create_dataset('discretization', data=np.asarray(self.delta))

                # save any fields
                fields = data.create_group('fields')
                for field in self.new_data:
                    fields.create_dataset(field, data=self.new_data[field][index])

                # add to the xdfm info
                xdfm.append_chrest_hdf5(hdf5_path)

        # write the xdmf file
        xdfm.write_to_file(str(path_template) + ".xdmf")


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
    field_data, times = chrest_data.get_field("flameTemp")

    # get the coordinate data
    coord_data = chrest_data.get_coordinates()

    # Print off a row of information at each time
    # for t in range(len(times)):
    t = 0
    time = chrest_data.times[t]
    i = 0
    j = 0
    k = 0
    for i in range(chrest_data.grid[0]):
        print(time, " @", coord_data[k, j, i, 0], ", ", coord_data[k, j, i, 1], ", ", coord_data[k, j, i, 2], ": ",
              field_data[t, k, j, i])
