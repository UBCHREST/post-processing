import argparse
import pathlib
import sys
import numpy as np
import h5py

from xdmfGenerator import XdmfGenerator
from supportPaths import expand_path


class ChrestData:
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

        # Create the expected variables
        # store the files based upon time
        self.files_per_time = dict()

        # store the list of fields available
        self.fields = []

        # load the metadata from the first file
        self.metadata = dict()

        # store the grid information
        self.start_point = []
        self.end_point = []
        self.delta = []
        self.dimensions = 0
        self.grid = []

        # store the times available to the data
        self.times = []

        # store any newly created data
        self.new_data = dict()

        # store an array of components names
        self.new_data_component_names = dict()

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
            if len(hdf5_data.attrs['time'].shape) == 0:
                tt = hdf5_data.attrs['time']
            else:
                tt = hdf5_data.attrs['time'][0]
            self.files_per_time[tt] = file

            # Load in the metadata
            if ~ len(self.metadata):
                for (key, value) in hdf5_data.attrs.items():
                    if isinstance(value, str):
                        self.metadata[key] = value
                    elif h5py.check_string_dtype(value.dtype):
                        self.metadata[key] = value[0]

            # store the grid information, it is assumed to be the same for each file
            hdf5_grid = hdf5_data['grid']
            if len(hdf5_grid['start'].shape) > 1:
                self.start_point = hdf5_grid['start'][:, 0].tolist()
                self.end_point = hdf5_grid['end'][:, 0].tolist()
                self.delta = hdf5_grid['discretization'][:, 0].tolist()
            else:
                self.start_point = hdf5_grid['start'][:].tolist()
                self.end_point = hdf5_grid['end'][:].tolist()
                self.delta = hdf5_grid['discretization'][:].tolist()
            self.dimensions = len(self.start_point)
            self.grid = []

            for dim in range(self.dimensions):
                if self.delta[dim] <= 0.0:
                    self.grid.append(1)
                else:
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
            if self.delta[dim] <= 0.0:
                self.grid.append(1)
            else:
                self.grid.append(int((self.end_point[dim] - self.start_point[dim]) / self.delta[dim]))

        # if the dim is less than 3, add addition components
        for d in range(self.dimensions, 3):
            self.start_point.append(0)
            self.end_point.append(0)
            self.grid.append(1)
            self.delta.append(0)

    """
    Returns a copy of the chrest data with the same grid without any times/fields
    """

    def copy_grid(self):
        chrest_data_copy = ChrestData()
        chrest_data_copy.setup_new_grid(self.start_point, self.end_point, self.delta)
        return chrest_data_copy

    """
    Returns the field data for the specified time range as a single (numpy array) and times.
    The field should be [t, k, j, i]
    
    returns field, times, component_names
    """

    def get_field(self, field_name, min_time=-sys.float_info.max, max_time=sys.float_info.max):
        # check if the field is in new data
        if field_name in self.new_data.keys():
            # do a quick check for the default
            if min_time == -sys.float_info.max and max_time == sys.float_info.max:
                return self.new_data[field_name], self.times
            else:
                # extract the data at the specified times
                data = []
                field_times = []
                t = 0  # the time index
                for tt in self.times:
                    if min_time <= tt <= max_time:
                        field_times.append(tt)
                        # Load in the file
                        data.append(self.new_data[field_name][t, ...])
                    t += 1

                return np.stack(data), field_times, self.new_data_component_names.get(field_name, None)

        else:  # load in the data from the file
            # create a dictionary of times/data
            data = []
            field_times = []
            component_names = None
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

                            if 'components' in hdf5_field.attrs:
                                component_names = hdf5_field.attrs['components'].tolist()

                        except Exception as e:
                            raise Exception(
                                "Unable to open field " + field_name + ". Valid fields include: " + ','.join(
                                    self.fields) + ". " + str(e))

            return np.stack(data), field_times, component_names

    """
    Creates and returns a new field (numpy array) and times.
    The field should be [t, k, j, i, c]

    """

    def create_field(self, field_name, number_components=1, component_names=None):
        # ensure at least one time
        if len(self.times) == 0:
            self.times.append(0.0)

        # determine the shape
        shape = [len(self.times)]

        # add in each grid, note the reverse for k, j, i formatting
        grid = self.grid.copy()
        grid.reverse()
        shape.extend(grid)

        # this is where any components would be added
        if number_components > 1:
            shape.append(number_components)

        # create the new field
        data = np.zeros(tuple(shape))

        # store it for future output
        self.new_data[field_name] = data
        if component_names is not None:
            self.new_data_component_names[field_name] = component_names

        # add to the list of fields
        self.fields.append(field_name)

        return data, self.times

    """
    Returns an numpy array of coordinates
    [k, j, i, dim]
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
                    try:
                        data.attrs.create(key, self.metadata[key])
                    except (Exception,) as e:
                        print("Could not save metadata", e)
                data.attrs.create('time', self.times[index])

                # write the grid information
                grid = data.create_group('grid')
                grid.create_dataset('start', data=np.asarray(self.start_point))
                grid.create_dataset('end', data=np.asarray(self.end_point))
                grid.create_dataset('discretization', data=np.asarray(self.delta))

                # save any fields
                fields = data.create_group('fields')
                for field in self.new_data:
                    newField = fields.create_dataset(field, data=self.new_data[field][index], compression="gzip",
                                                     dtype=np.float32)
                    if field in self.new_data_component_names:
                        newField.attrs.create('components', self.new_data_component_names[field])

            # add to the xdfm info
            xdfm.append_chrest_hdf5(hdf5_path)

        # write the xdmf file
        xdfm.write_to_file(str(path_template) + ".xdmf")

    """
    Compute the mean and rms of a field and return in new chrest data
    """

    def compute_mean_rms(self, field_name, field_data):
        # create a copy to store statistics
        statistics_data = self.copy_grid()

        # example to take avg rms of temperature
        mean_field = field_data.mean(axis=0)
        statistics_data.create_field(field_name + '_mean')[0][0] = mean_field

        rms_field = statistics_data.create_field(field_name + '_rms')[0][0]
        for t in range(field_data.shape[0]):
            rms_field[:] = np.add(rms_field[:], np.square(field_data[t, ...]))

        rms_field[:] = rms_field[:] / field_data.shape[0] - np.square(mean_field[:])
        rms_field[:] = np.maximum(rms_field[:], 0)
        rms_field[:] = np.sqrt(rms_field[:])

        return statistics_data

    """
    Compute the mean and rms of a field and return in new chrest data
    """

    def compute_mean_rms_field_name(self, field_name):
        # load in an example data
        field_data = self.get_field(field_name)[0]

        return self.compute_mean_rms(field_name, field_data)


# parse based upon the supplied inputs
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate an xdmf file from MatLab data files holding structured data. '
                    'See https://github.com/cet-lab/experimental-post-processing/wiki'
                    '/Matlab-To-XdmfGenerator for details.  ')
    parser.add_argument('--file', dest='hdf5_file', type=pathlib.Path, required=True,
                        help='The path to the hdf5 file(s) containing the structured data.  A wild card can be used '
                             'to supply more than one file.')
    parser.add_argument('--stats', dest='stats_file', type=pathlib.Path,
                        help='Path to write stats file')
    parser.add_argument('--field', dest='field', type=str,
                        help='Path to the stats file', required=True, )
    parser.add_argument('--example', dest='example', action='store_true',
                        help='If true, runs through access example', default=False)
    args = parser.parse_args()

    # this is some example code for chest file post processing
    chrest_data = ChrestData(args.hdf5_file)

    # compute the rms/mean
    statistics_data = chrest_data.compute_mean_rms_field_name(args.field)

    if args.stats_file is not None:
        statistics_data.save(args.stats_file)

    if args.example:
        # this is an example of accessing the resulted statistics_data
        print("Fields: ", statistics_data.fields)

        # get the rms of the field
        rms_field, times, componentsNames = statistics_data.get_field(args.field + "_rms", -1.0)

        print(args.field + "_rms shape: ", rms_field.shape)
        print(args.field + "_rms times: ", times)
