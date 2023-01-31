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
    filesPerTime = dict()

    # store the list of fields available
    fields = []

    # load the metadata from the first file
    metadata = dict()

    # store the grid information
    startPoint = []
    endPoint = []
    delta = []
    dimensions = 0
    grid = []

    """
    Creates a new class from hdf5 chrest formatted file(s).
    """

    def __init__(self, files):
        if isinstance(files, str):
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
            time = hdf5_data.attrs['time'][0]
            self.filesPerTime[time] = file

            # Load in the metadata
            if ~ len(self.metadata):
                for (key, value) in hdf5_data.attrs.items():
                    if h5py.check_string_dtype(value.dtype):
                        self.metadata[key] = value[0]

            # store the grid information, it is assumed to be the same for each file
            hdf5_grid = hdf5_data['grid']
            self.startPoint = hdf5_grid['start'][:, 0].tolist()
            self.endPoint = hdf5_grid['end'][:, 0].tolist()
            self.delta = hdf5_grid['discretization'][:, 0].tolist()
            self.dimensions = len(self.startPoint)

            for dim in range(self.dimensions):
                self.grid.append(int((self.endPoint[dim] - self.startPoint[dim]) / self.delta[dim]))

    """
    Returns the field data for the specified time range as a list of pairs(time, numpy array).
    The field should be [k, j, i]
    
    """

    def GetField(self, field_name, min_time=-sys.float_info.max, max_time=sys.float_info.max):
        # create a dictionary of times/data
        data = []

        # march over the files
        times = list(self.filesPerTime.keys())
        times.sort()
        for time in times:
            if min_time <= time <= max_time:
                # Load in the file
                with h5py.File(self.filesPerTime[time], 'r') as hdf5:
                    # Check each of the cell fields
                    hdf5_fields = hdf5['data/fields']

                    try:
                        # load in the specified field name
                        hdf5_field = hdf5_fields[field_name]

                        data.append((time, hdf5_field[:]))
                    except Exception as e:
                        raise Exception(
                            "Unable to open field " + field_name + ". Valid fields include: " + ','.join(self.fields))

        return data

    """
    Returns an numpy array of coordinates
    [dim][k][j][i]
    """

    def GetCoordinates(self):
        # Note we reverse the order of the linespaces and the returning grid for k,j,i index
        # create a list of line spaces
        linspaces = []
        for dim in range(self.dimensions):
            linspaces.append(
                np.linspace(self.startPoint[dim] + self.delta[dim] / 2.0, self.endPoint[dim] - self.delta[dim] / 2.0,
                            self.grid[dim], endpoint=True))

        linspaces.reverse()

        # compute the multi dim grid
        grid = np.meshgrid(*linspaces, indexing='ij')
        grid.reverse()
        return grid


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
    field_data = chrest_data.GetField("flameTemp")

    # get the coordinate data
    coord_data = chrest_data.GetCoordinates()

    # Print off a row of information at each time
    for (time, flame_temp) in field_data:
        i = 0
        j = 0
        k = 0
        for i in range(chrest_data.grid[0]):
            print(time, " @", coord_data[0][k, j, i], ", ", coord_data[1][k, j, i], ", ", coord_data[2][k, j, i], ": ",
                  flame_temp[k, j, i])
